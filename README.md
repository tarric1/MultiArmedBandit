# Il bandito bracciuto
## Introduzione leggera ma non troppo all'apprendimento con rinforzo.

Siete stanchi di lavorare e volete passare una vita in vacanza?

Bene, andiamo al casinò, però non siamo degli sprovveduti, quindi dobbiamo scegliere con attenzione la nostra strategia: a cosa giochiamo?

Alla roulette?

Non sembra una buona idea, al tavolo c'è quel signore pelato che sta vincendo tutto!

Giochiamo a poker?

Ehi... ma non sappiamo giocare a poker!

Giochiamo alle slot machine!

Siamo andati in un casinò retrò, quindi le slot machine sono quelle meccaniche a 3 rulli con 10 simboli ciascuno e con una leva, il braccio, in inglese _arm_, che bisogna tirare per far ruotare i rulli. Si vince quando su tutti rulli compare lo stesso simbolo. Che probabilità abbiamo di vincere?

$$
\large p_{win}=\frac{1}{10^{3}}=0.001
$$

Un po' bassa, non per nulla le chiamano _macchinette mangia soldi_ ed i nostri amici inglesi le chiamano _bandit_, _armed bandit_ per la precisione, letteralmente _bandito bracciuto_, facendo riferimento alla leva che le aziona.

Va bene, non disperiamoci, nel casinò c'è una file di slot machine, sono ancora quelle meccaniche e la probabilità di vincere non è certamente quella teorica che abbiamo calcolato.

Come facciamo ad individuare la slot machine con la probabilità di vincere a noi più favorevole?

Dobbiamo provare a giocare con tutte, registriamo per ogni slot machine:

- il numero di volte che abbiamo vinto;
- il numero di volte che abbiamo giocato;

il rapporto ci fornisce una stima della probabilità di vincere:

$$
\large \tilde{p}_{win_{i}}=\frac{n_{win_{i}}}{n_{i}}
$$

Facciamo una simulazione adoperando Python.

Tanto per cominciare dobbiamo modellare il bandito, la slot machine:

```python
import numpy as np

class Bandit:
    def __init__(self, reels: int, symbols: int):
        self.pwin: float = 1 / np.power(symbols, reels)

    def interact(self) -> float:
        return 1 if np.random.rand() < self.pwin else 0
```

Vediamo com'è fatta la classe ```Bandit```.

Il costruttore accetta come parametri di input:
- ```reels```: numero di rulli;
- ```symbols```: numero di simboli presenti su ciascun rullo;

e calcola la probabilità di vincere ```pwin```.

Il metodo ```interact()``` simula l'azione di giocare alla slot machine, per cui genera un numero casuale uniformemente distribuito fra 0 ed 1, se è minore della probabilità di vincere, abbiamo vinto ed il metodo restituisce 1, altrimenti restituisce 0.

Abbiamo detto però, che le slot machine sono meccaniche, per cui la probabilità di vincere non è quella teorica, ma non può allontanarsi molto da questa, per cui aggiungiamo al costruttore un'ulteriore parametro ```delta``` che indica questo scostamento:

```python
import numpy as np

class Bandit:
    def __init__(self, reels: int, symbols: int, delta: float):
        self.pwin: float = 1 / np.power(symbols, reels) + delta

    def interact(self) -> float:
        return 1 if np.random.rand() < self.pwin else 0
```

Adesso dobbiamo modellare il giocatore, che gioca alle slot machine e registra le vincite per stimare la probabilità di vincita:

```python
from base.Bandit import Bandit

class Agent:
    def __init__(self, bandit: Bandit):
        self.bandit: Bandit = bandit
        self.k: int = 0
        self.q: float = 0

    def do(self) -> float:
        return self.bandit.interact()

    def update(self, r: float):
        self.q += (r - self.q) / (self.k + 1)
        self.k += 1

    def play(self) -> int:
        r: float = self.do()
        self.update(r)
        return r
```

La classe ```Agent``` ha tre attributi:

- ```bandit```: la slot machine con cui giocare, ossia il puntatore all'istanza della classe ```Bandit``` che simula la slot machine;
- ```k```: numero di volte che il giocatore ha giocato;
- ```q```: probabilità stimata di vincere.

Gli ultimi due attributi corrispondono rispettivamente a:

$$
\large n_{i}, \tilde{p}_{win_{i}}
$$

Inoltre la classe ha tre metodi:

- ```do()```: simula l'azione di giocare alla slot machine, per cui richiama il metodo ```interact()``` dell'istanza della ```Bandit```; il metodo restituisce l'esito della giocata, per cui 1 se si vince, 0 se si perde;
- ```update(float)```: aggiorna il numero di volte che il giocatore ha giocato ```k``` e la probabilità stimata di vincere ```q```, in base all'esito ```r``` della giocata;
- ```play()```: richiama in cascata i metodo ```do()``` e ```update(float)``` passando a quest'ultimo l'esito della giocata ottenuto dall'invocazione del primo metodo.  

Facciamo un'esperimento con le due classi:

```python
from typing import List
from base.Bandit import Bandit
from base.Agent import Agent
import numpy as np
import matplotlib.pyplot as plt

class Experiment:
    @staticmethod
    def main():
        reels: int = 3
        symbols: int = 10
        delta: float = 0.0002
        n: int = 1000000

        bandit: Bandit = Bandit(reels, symbols, delta)
        agent: Agent = Agent(bandit)

        r: List[float] = [0] * n
        for i in range(n):
            r[i] = agent.play()

        rewards_trend: List[float] = np.cumsum(r) / np.arange(1, n + 1)
        plt.plot(rewards_trend, label='mean reward = {0:.5f}'.format(rewards_trend[-1]))

        plt.plot([0, n - 1], [bandit.pwin, bandit.pwin],
                 label='pwin = {0:.5f}'.format(bandit.pwin))
        print('q = {0:.5f}  pwin = {1:.5f}'.format(agent.q, bandit.pwin))

        plt.xscale('linear')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    Experiment.main()
```

La classe ```Experiment``` ha il metodo statico ```main``` che crea l'oggetto rappresentate la slot machine, quello rappresentante il giocatore, quindi esegue 1000000 di interazioni e per concludere visualizza l'andamento della probabilità stimata di vincere:

![](first_test.svg)

```
q = 0.00121  pwin = 0.00120
```

Ok, funziona ma non dimentichiamo che l'obiettivo è modellare il giocatore in modo che possa individuare  la slot machine con la probabilità di vincere più favorevole, per cui modifichiamo la classe ```Agent```: 

```python
from typing import List
from base.Bandit import Bandit
import numpy as np

class Agent:
    def __init__(self, bandits: List[Bandit]):
        self.bandits: List[Bandit] = bandits
        self.n: int = len(bandits)
        self.k: List[int] = [0] * self.n
        self.q: List[float] = [0] * self.n

    def choose(self) -> int:
        return np.random.choice(self.n)

    def do(self, a: int) -> float:
        return self.bandits[a].interact()

    def update(self, a: int, r: float):
        self.q[a] += (r - self.q[a]) / (self.k[a] + 1)
        self.k[a] += 1

    def play(self) -> int:
        a: int = self.choose()
        r: float = self.do(a)
        self.update(a, r)
        return r
```

L'attributo ```bandit``` ha lasciato il posto a ```bandits``` che contiene la lista delle slot machine cui giocare, mentre gli attributi ```k``` e ```q``` sono diventati anch'essi delle liste contenenti per ogni slot machine il numero di volte che il giocatore ha giocato e la probabilità stimata di vincere.

E' stato aggiunto il metodo ```choose()``` che in maniera casuale con una distribuzione uniforme sceglie la slot machine cui giocare.

I metodi ```do(int)``` ed ```update(int, float)``` sono stati modificati in modo da accettare come parametro di input l'indicazione della slot machine cui giocare, mentre il metodo ```play()``` prima di chiamare in sequenza i due metodi, invoca ```choose()```.

Facciamo un altro esperimento e verifichiamo che la classe ```Agent``` riesca ad individuare la slot machine con la probabilità di vincere più favorevole:

```python
from typing import List
from base.Bandit import Bandit
from randompolicy.Agent import Agent
import numpy as np
import matplotlib.pyplot as plt

class Experiment:
    @staticmethod
    def main():
        reels: int = 3
        symbols: int = 10
        deltas: List[float] = [0.0002, 0.0001, -0.0001]
        n: int = 1000000

        bandits: List[Bandit] = [Bandit(reels, symbols, delta) for delta in deltas]
        agent: Agent = Agent(bandits)

        r: List[float] = [0] * n
        for i in range(n):
            r[i] = agent.play()

        rewards_trend: List[float] = np.cumsum(r) / np.arange(1, n + 1)
        plt.plot(rewards_trend, label='mean reward = {0:.5f}'.format(rewards_trend[-1]))

        for i in range(len(bandits)):
            plt.plot([0, n - 1], [bandits[i].pwin, bandits[i].pwin], label='pwin = {0:.5f}'.format(bandits[i].pwin))
            print('Bandit #{0} : q = {1:.5f}  pwin = {2:.5f}'.format(i, agent.q[i], bandits[i].pwin))

        plt.xscale('linear')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    Experiment.main()
```




### APPUNTI  ADDESTRAMENTO  CON  RINFORZO

Serve per permettere ad un <u>agente</u> di agire in un <u>ambiente</u>.

Quando l'agente compie delle azioni nell'ambiente, riceve dei <u>feedback</u> (<u>ricompense</u>) dall'ambiente. Il feedback può essere anche negativo, in questo caso piuttosto che di ricompensa si dovrebbe parlare di <u>penalità</u>, però se consideriamo la penalità come una <u>ricompensa negativa</u>, possiamo continuare ad usare il termine ricompensa.

L'agente deve raggiungere un <u>obiettivo</u> (<u>goal</u>).

Le diverse <u>configurazioni</u> nelle quali può trovarsi l'ambiente costituiscono gli <u>stati</u> dell'ambiente.

L'obiettivo dell'agente è <u>massimizzare le ricompense future</u> e non solo quelle immediate (<i>spiegare meglio</i>).

<u>Azione</u>: E' ciò che l'agente può fare nell'ambiente.

**L'ambiente di trova in uno stato s(t), l'agente compie un'azione a(t), questa fa passare l'ambiente nello stato s(t+1) e l'agente riceve una ricompensa (che può essere negativa) r(t+1).**

