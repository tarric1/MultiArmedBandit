# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 11:49:14 2018

Experiment.py

@author: Fabrizio A. Tarricone
"""

import numpy as np
from optimisticInitialValues.Bandit import Bandit


class Experiment:
    # real_means : Lista contenente le medie vere su cui svolgere l'esperimento; rappresenta le slot machine che
    #              l'ipotetico giocatore può usare e sulle quali si pone il dilemma exploration/exploitation.
    # upper_limit: Ipotetico valore massimo del valore medio delle ricompense.
    # n          : Numero totale di giocate da eseguire durante l'esperimento.
    @staticmethod
    def run(real_means, upper_limit, n):
        # Le slot machine (i banditi)...
        bandits = [Bandit(real_mean, upper_limit) for real_mean in real_means]

        rewards = np.empty(n)

        for i in range(n):
            # Optimistic Initial Value.
            j = np.argmax([bandit.estimated_mean for bandit in bandits])

            reward = bandits[j].pull()
            bandits[j].update(reward)

            rewards[i] = reward

        # Calcoliamo la ricompensa media ottenuta durante l'avanzare delle giocate.
        mean_rewards = np.cumsum(rewards) / np.arange(1, n + 1)
        return mean_rewards
