#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 Ben Lindsay <benjlindsay@gmail.com>

import numpy as np
import pandas as pd

class GC_Diblock_Homopolymer():
    def __init__(self, z_D, z_HA, N, f, alpha, rho_0, chi, kappa, **kwargs):
        self.z_D = z_D
        self.z_HA = z_HA
        self.N = N
        self.f = f
        self.alpha = alpha
        self.rho_0 = rho_0
        self.chi = chi
        self.kappa = kappa
        self.i_wpl = kwargs.get('i_wpl', 0)
        self.i_wabp = kwargs.get('i_wabp', 0)
        self.wabm = kwargs.get('wabm', 0.1)
        self.tol = kwargs.get('tol', 1e-10)
        self.lam_pl = kwargs.get('lam_pl', 0.01)
        self.lam_mi = kwargs.get('lam_mi', 0.005)
        self.save_freq = kwargs.get('save_freq', 20)
        self.iter = 0
        self.max_iter = kwargs.get('max_iter', 10000)
        self.update_densities()
        self.update_H_over_V()
        self.columns = ['iter', 'i_wpl', 'i_wabp', 'wabm', 'rho_D', 'rho_HA', 'H_over_V']
        row_0 = [self.iter, self.i_wpl, self.i_wabp, self.wabm, self.rho_D, self.rho_HA, self.H_over_V]
        self.df = pd.DataFrame([row_0], columns=self.columns)
        self.df_err = pd.DataFrame(columns=self.columns)
    def calc_i_dhdwpl(self):
        return self.rho_0 / self.kappa * self.i_wpl + self.rho_0 - self.rho_D - self.rho_HA
    def calc_i_dhdwabp(self):
        return 2 * self.rho_0 * self.i_wabp / self.chi - self.rho_D - self.rho_HA
    def calc_dhdwabm(self):
        return 2 * self.rho_0 * self.wabm / self.chi + (1 - 2*self.f) * self.rho_D - self.rho_HA
    def update_fields(self):
        self.i_wpl -= self.lam_pl * self.calc_i_dhdwpl()
        self.i_wabp -= self.lam_pl * self.calc_i_dhdwabp()
        self.wabm -= self.lam_mi * self.calc_dhdwabm()
        # self.i_wpl = self.kappa / self.rho_0 * (self.rho_D + self.rho_HA - self.rho_0)
        # self.i_wabp = self.chi / (2 * self.rho_0) * (self.rho_D + self.rho_HA)
        # self.wabm = self.chi / (2 * self.rho_0) * ((2*self.f - 1) * self.rho_D + self.rho_HA)
    def update_densities(self):
        self.rho_D = self.z_D * self.N * np.exp( - self.N * (self.i_wpl + self.i_wabp)
                                                 + (2*self.f - 1) * self.N * self.wabm )
        self.rho_HA = ( self.z_HA * self.alpha * self.N
                        * np.exp(-self.alpha * self.N * (self.i_wpl + self.i_wabp - self.wabm))
                      )
    def update_H_over_V(self):
        H_over_V = - self.rho_0 / (2 * self.kappa) * self.i_wpl ** 2 - self.rho_0 * self.i_wpl 
        H_over_V += - self.rho_0 / self.chi * self.i_wabp ** 2
        H_over_V += self.rho_0 / self.chi * self.wabm ** 2
        H_over_V += - self.rho_D / self.N - self.rho_HA / (self.alpha * self.N)
        self.H_over_V = H_over_V
    def update_dfs(self):
        new_row = [self.iter, self.i_wpl, self.i_wabp, self.wabm, self.rho_D, self.rho_HA, self.H_over_V]
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
