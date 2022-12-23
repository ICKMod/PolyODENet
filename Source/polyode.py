import string
from itertools import product

import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint


class PolynomialODE(nn.Module):
    def __init__(self, indices, scaler, guess,
                 n_max_reaction_order=2, n_species=3,
                 include_zeroth_order=False, include_self_reaction=False,
                 device=torch.device('cpu')):
        super(PolynomialODE, self).__init__()
        self.n_max_reaction_order = n_max_reaction_order
        self.n_species = n_species
        ei = list(product(range(n_max_reaction_order + 1),
                          repeat=n_species))  # enumerate all possible concentration exponent combinations
        ei = [x for x in ei if sum(x) <= n_max_reaction_order]
        ei = sorted(ei, reverse=True,
                    key=lambda x: (sum(np.array(x) ** 1.1), x))  # high order first, keep species order
        assert ei[-1] == (0,) * n_species
        if not include_zeroth_order:
            ei.remove((0,) * n_species)
        #  Remove self reaction terms for 2nd order
        if n_max_reaction_order == 2 and not include_self_reaction:
            del ei[:n_species]
        self.exponent_indices = np.array(ei, dtype='int')
        self.species_indices = np.arange(n_species). \
            reshape([1, n_species]). \
            repeat(self.exponent_indices.shape[0], axis=0)
        bw = torch.zeros(self.exponent_indices.shape[0], n_species,
                         dtype=torch.float64, requires_grad=True, device=device)
        self._basis_weights = nn.Parameter(bw, requires_grad=True)
        if guess.size > 0:
            self._basis_weights.data = torch.tensor(guess, dtype=torch.float64, requires_grad=True, device=device)
        else:
            torch.nn.init.xavier_normal_(self._basis_weights, gain=0.5)
            # torch.nn.init.sparse_(self.basis_weights, sparsity=0.75, std=0.1)
        if indices.size > 0:
            self.row_indices = indices[:, 0::2]
            self.col_indices = indices[:, 1::2]
        else:
            self.row_indices = np.arange(self.exponent_indices.shape[0]). \
                reshape(self.exponent_indices.shape[0], 1). \
                repeat(self.exponent_indices.shape[1], axis=1)
            self.col_indices = self.species_indices
        if scaler.size > 0:
            self.scaler = torch.tensor(scaler, dtype=torch.float64, device=device)
        else:
            self.scaler = torch.ones(self.exponent_indices.shape, dtype=torch.float64, device=device)

    @property
    def basis_weights(self):
        return self._basis_weights[self.row_indices, self.col_indices] * self.scaler

    def forward(self, _, conc_in):
        # calculate the reaction at the given concentrations
        conc_power_item = torch.stack([conc_in ** e
                                       for e in range(self.n_max_reaction_order + 1)],
                                      dim=-1)
        conc_power = conc_power_item[...,
                                     self.species_indices,
                                     self.exponent_indices].prod(dim=-1)
        reaction_rate = conc_power @ self.basis_weights  # matrix product
        return reaction_rate

    def __repr__(self):
        # print the reaction rate equations
        parent_name = super().__repr__()
        lines = [f'{parent_name}:']
        species_names = string.ascii_uppercase

        def name_prod_item(i_species, exponent):
            if exponent == 0:
                return ''
            elif exponent == 1:
                return f'[{species_names[i_species]}]'
            else:
                return f'[{species_names[i_species]}]^{exponent}'

        conc_prod_names = [''.join([name_prod_item(i_species, e)
                                    for i_species, e in zip(range(self.n_species), exponents)])
                           for exponents in self.exponent_indices]
        basis_weights = self.basis_weights.detach().clone().cpu().numpy()
        for i_species in range(self.n_species):
            species_text_list = []
            for j_basis in range(self.basis_weights.shape[0]):
                pn = conc_prod_names[j_basis]
                bw = basis_weights[j_basis, i_species]
                if bw != 0:
                    item_text = f'{bw:+.3G}{pn}'
                    item_text = item_text[0] + ' ' + item_text[1:]
                    species_text_list.append(item_text)
            species_rate_equation = ' '.join(species_text_list)
            if len(species_rate_equation) == 0:
                species_rate_equation = '0'
            elif species_rate_equation[0] == '+':
                species_rate_equation = species_rate_equation[2:]
            else:
                species_rate_equation = species_rate_equation[0] + species_rate_equation[2:]
            lines.append(f'd[{species_names[i_species]}]/dt = {species_rate_equation}')
        return '\n\n'.join(lines)

    @staticmethod
    def get_batch(concentrations, timestamps, full_indices, n_segments, segment_len, device):
        # segment_len will be the batch size
        candidate_indices = full_indices[:-segment_len]  # leave enough space for a full segment
        replace = n_segments > candidate_indices.shape[0]
        seg_start_indices = np.random.choice(candidate_indices, n_segments, replace=replace)  # (n_segments)
        segment_indices = seg_start_indices.reshape([-1, 1]). \
            repeat(segment_len, axis=1)  # (n_segments, segment_len), identical indices
        segment_indices += np.arange(segment_len)  # (n_segments, segment_len), incremental indices

        segment_indices = torch.tensor(segment_indices, device=device)

        batch_t = timestamps[segment_indices.flatten()].view(*segment_indices.size())  # (n_segments, n_species)
        batch_y = concentrations[segment_indices.flatten(), :] \
            .view(*segment_indices.size(), concentrations.size()[-1])  # (n_segments, segment_len, n_species)
        batch_y0 = batch_y[:, 0, :]  # (n_segments, n_species)

        return batch_y0, batch_t, batch_y

    @staticmethod
    def batch_odeint(func, batch_y0, batch_t):
        pred_y = torch.stack([odeint(func, by0, bt) for by0, bt in zip(batch_y0, batch_t)], dim=0)
        return pred_y
