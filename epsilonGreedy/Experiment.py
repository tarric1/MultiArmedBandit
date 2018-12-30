# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 11:49:14 2018

Experiment.py

@author: Fabrizio A. Tarricone
"""

import numpy as np
from epsilonGreedy.Bandit import Bandit


class Experiment:
    # real_means: Lista contenente le medie vere su cui svolgere l'esperimento; rappresenta le slot machine che
    #             l'ipotetico giocatore può usare e sulle quali si pone il dilemma exploration/exploitation.
    # epsilon   : Valore di epsilon da adoperare per l'esperimento; indica la percentuale di giocate per le quali si
    #             esplora, ossia si cambia slot machine.
    # n         : Numero totale di giocate da eseguire durante l'esperimento.
    @staticmethod
    def run(real_means, epsilon, n):
        # Numero di slot machine.
        m = len(real_means)

        # Le slot machine (i banditi)...
        bandits = [Bandit(real_mean) for real_mean in real_means]

        rewards = np.empty(n)

        for i in range(n):
            # Epsilon-Greedy.
            #
            # Distribuiamo le giocate fra le slot machine, in modo che per una percentuale di giocate pari epsilon
            # cambiano la slot machine scegliendone una a caso (exploration) e per la parte restante delle giocate
            # adoperiamo la slot machine che ha, in base ai dati raccolti durante le giocate, il maggiore valor medio
            # stimato delle ricompense (exploitation).
            p = np.random.random()
            if p < epsilon:
                # Cambiamo slot machine (exploration).
                j = np.random.choice(m)
            else:
                # Adoperiamo la slot machine con la media stimata maggiore, ossia quella che secondo i dati raccolti
                # durante le giocate fornisce le ricompense maggiori (exploitation).
                j = np.argmax([bandit.estimated_mean for bandit in bandits])

            reward = bandits[j].pull()
            bandits[j].update(reward)

            rewards[i] = reward

        # Calcoliamo la ricompensa media ottenuta durante l'avanzare delle giocate.
        mean_rewards = np.cumsum(rewards) / np.arange(1, n + 1)
        return mean_rewards
