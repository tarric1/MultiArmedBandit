# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 11:40:00 2018

Questa classe modella il bandito che raffigura la slot machine.

@author: Fabrizio A. Tarricone
"""

import numpy as np


class Bandit:
    # Costruttore.
    #
    # real_mean  : Media vera.
    # upper_limit: Ipotetico valore massimo della media vera; viene adoperato per inizializzare la media stimata.
    def __init__(self, real_mean, upper_limit):
        self.real_mean = real_mean  # Media vera.
        self.estimated_mean = upper_limit  # Media stimata.
        self.n = 0  # Numero di giocatre effettuate.

    # Il metodo viene richiamato quando si simula una giocata e restituisce il valore della ricomenpsa.
    def pull(self):
        # La ricompensa ha un valore casuale avente distribuzione gaussiana con varianza pari a 1 e media pari alla
        # media vera real_mean.
        return np.random.randn() + self.real_mean

    # Il metodo viene richiamato dopo aver eseguito una giocata.
    #
    # Il metodo aggiorna il numero di giocate ed il valore della media stimata.
    #
    # reward: Ricompensa ottenuta nell'ultima giocata.
    def update(self, reward):
        m = self.n + 1
        # La media stimata viene calcolata partendo da quella stimata nella giocata precedente.
        self.estimated_mean = (self.n * self.estimated_mean + reward) / m
        self.n = m
