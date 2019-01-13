# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 11:49:14 2018

Experiment.py

@author: Fabrizio A. Tarricone
"""

import numpy as np
from src.ucb1.Bandit import Bandit


class Experiment:
    # real_means : Lista contenente le medie vere su cui svolgere l'esperimento; rappresenta le slot machine che
    #              l'ipotetico giocatore può usare e sulle quali si pone il dilemma exploration/exploitation.
    # n          : Numero totale di giocate da eseguire durante l'esperimento.
    @staticmethod
    def run(real_means, n):
        # Le slot machine (i banditi)...
        bandits = [Bandit(real_mean) for real_mean in real_means]

        rewards = np.empty(n)

        for i in range(n):
            # Qual è l'intuizione?
            #
            # Il metodo Optimistic Initial Value ha il problema di dover "tirare ad indovinare" il limite massimo
            # (upper_limit) delle medie vere dei bandit.
            #
            # Il problema è che se il valore impostato è errato, si corre il rischio di privilegiare le giocate sul
            # bandit errato, in quando l'exploration protrebbe fermarsi prematuramente.
            #
            # Per forzare l'explorarion, invece di adoperare un upper_limit che si tenta di indovinare, si adopera
            # una formula che genera un valore che aumenta quando il numero di giocate su un bandit è basso; in questo
            # modo l'algortimo sceglie i bandit che hanno poche giocate.
            #
            # In realtà la componente greedy dell'algortimo resta in essere, per cui nella scelta si considera anche il
            # valore stima della ricompensa media.
            #
            # In questo modo si bilacia exploitation (greedy) e l'exploration (bandit con poche giocate).
            #
            # NOTA: bandit.n + 0.001 per evitare la divisione per 0.
            j = np.argmax([(bandit.estimated_mean + np.sqrt(2 * np.log(i + 1) / (bandit.n + 0.001))) for bandit in bandits])

            reward = bandits[j].pull()
            bandits[j].update(reward)

            rewards[i] = reward

        # Calcoliamo la ricompensa media ottenuta durante l'avanzare delle giocate.
        mean_rewards = np.cumsum(rewards) / np.arange(1, n + 1)
        return mean_rewards
