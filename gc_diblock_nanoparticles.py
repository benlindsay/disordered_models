#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 Ben Lindsay <benjlindsay@gmail.com>

import numpy as np
import pandas as pd
import scipy.special as ss

class GC_Diblock_Nanoparticles():
    def __init__(self, z_D, z_P, N, f, R_P, xi_P, rho_0, chi, kappa, **kwargs):
        self.z_D = z_D
        self.z_P = z_P
        self.N = N
        self.f = f
        self.R_P = R_P
        self.xi_P = xi_P
        self.rho_0 = rho_0
        self.chi = chi
        self.kappa = kappa
        self.dim = kwargs.get('dim', 3)
        self.i_wpl = kwargs.get('i_wpl', 0)
        self.i_wabp = kwargs.get('i_wabp', 0)
        self.wabm = kwargs.get('wabm', 0.1)
        self.tol = kwargs.get('tol', 1e-10)
        self.lam_pl = kwargs.get('lam_pl', 0.005)
        self.lam_mi = kwargs.get('lam_mi', 0.0025)
        self.save_freq = kwargs.get('save_freq', 20)
        self.iter = 0
        self.max_iter = kwargs.get('max_iter', 10000)
        self.V_P = self.calc_V_P()
        self.update_densities()
        self.update_H_over_V()
        self.columns = ['iter', 'i_wpl', 'i_wabp', 'wabm', 'rho_D', 'rho_P', 'phi_P', 'H_over_V']
        row_0 = [self.iter, self.i_wpl, self.i_wabp, self.wabm, self.rho_D, self.rho_P, self.phi_P, self.H_over_V]
        self.df = pd.DataFrame([row_0], columns=self.columns)
        self.df_err = pd.DataFrame(columns=self.columns)
    def calc_V_P(self):
        L_r = 5 * self.R_P
        n_pts = int(10000 * L_r / self.xi_P)
        dr = L_r / n_pts
        r = np.linspace(0, L_r, n_pts)
        phi_P = 0.5 * ss.erfc( (r - self.R_P) / self.xi_P )
        if self.dim == 2:
            return np.sum(2 * np.pi * r * phi_P * dr)
        elif self.dim == 3:
            return np.sum(4 * np.pi * r**2 * phi_P * dr)
    def calc_i_dhdwpl(self):
        return self.rho_0 / self.kappa * self.i_wpl + self.rho_0 - self.rho_D - self.rho_P
    def calc_i_dhdwabp(self):
        return 2 * self.rho_0 * self.i_wabp / self.chi - self.rho_D - self.rho_P
    def calc_dhdwabm(self):
        return 2 * self.rho_0 * self.wabm / self.chi + (1 - 2*self.f) * self.rho_D - self.rho_P
    def update_fields(self):
        self.i_wpl -= self.lam_pl * self.calc_i_dhdwpl()
        self.i_wabp -= self.lam_pl * self.calc_i_dhdwabp()
        self.wabm -= self.lam_mi * self.calc_dhdwabm()
    def update_densities(self):
        self.rho_D = self.z_D * self.N * np.exp( - self.N * (self.i_wpl + self.i_wabp)
                                                 + (2*self.f - 1) * self.N * self.wabm )
        self.rho_P = ( self.z_P * self.rho_0 * self.V_P
                        * np.exp(-self.rho_0 * self.V_P * (self.i_wpl + self.i_wabp - self.wabm))
                      )
        self.phi_P = self.rho_P / (self.rho_D + self.rho_P)
    def update_H_over_V(self):
        H_over_V = - self.rho_0 / (2 * self.kappa) * self.i_wpl ** 2
        H_over_V += - self.rho_0 * self.i_wpl
        H_over_V += - self.rho_0 / self.chi * self.i_wabp ** 2
        H_over_V += self.rho_0 / self.chi * self.wabm ** 2
        H_over_V += - self.rho_D / self.N
        H_over_V += - self.rho_P / (self.rho_0 * self.V_P)
        self.H_over_V = H_over_V
    def update_dfs(self):
        new_row = [self.iter, self.i_wpl, self.i_wabp, self.wabm, self.rho_D, self.rho_P, self.phi_P, self.H_over_V]
        self.df = self.df.append(pd.DataFrame([new_row], columns=self.columns), ignore_index=True)
        new_df_err_row = np.abs((self.df.iloc[-1, 1:] - self.df.iloc[-2, 1:]) / self.df.iloc[-2, 1:])
        new_df_err_row['iter'] = self.iter
        self.df_err = self.df_err.append(new_df_err_row, ignore_index=True)
    def run(self):
        for self.iter in range(1, self.max_iter+1):
            self.update_fields()
            self.update_densities()
            self.update_H_over_V()
            if self.iter % self.save_freq == 0:
                self.update_dfs()
                if self.df_err.iloc[-1, 1:].max() < self.tol:
                    break
