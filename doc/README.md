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

    def update(self, r: float):
        self.q += (r - self.q) / (self.k + 1)
        self.k += 1

    def do(self) -> float:
        r: float = self.bandit.interact()
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

- ```update(float)```: simula l'operazione di registrazione dei dati, ossia aggiorna il numero di volte ```k``` che il giocatore ha giocato e la probabilità stimata di vincere ```q```, in base all'esito ```r``` della giocata;
- ```do()```: simula l'azione di giocare alla slot machine, per cui richiama il metodo ```interact()``` dell'istanza di ```Bandit```, e di registrazione dei dati tramite il metodo ```update(float)```.  

Facciamo un esperimento con le due classi:

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
            r[i] = agent.do()

        rewards_trend: List[float] = np.cumsum(r) / np.arange(1, n + 1)
        plt.plot(rewards_trend, label='mean reward = {0:.5f}'.format(rewards_trend[-1]))

        plt.plot([0, n - 1], [bandit.pwin, bandit.pwin], label='pwin = {0:.5f}'.format(bandit.pwin))
        print('q = {0:.5f}  pwin = {1:.5f}'.format(agent.q, bandit.pwin))

        plt.xscale('log')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    Experiment.main()
```

La classe ```Experiment``` ha il metodo statico ```main``` che crea l'oggetto rappresentate la slot machine, quello rappresentante il giocatore, quindi esegue 1000000 di interazioni e per concludere visualizza l'andamento della probabilità stimata di vincere:

![](first_test.png)

```
q = 0.00115  pwin = 0.00120
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

    def update(self, a: int, r: float):
        self.q[a] += (r - self.q[a]) / (self.k[a] + 1)
        self.k[a] += 1

    def do(self) -> float:
        a: int = self.choose()
        r: float = self.bandits[a].interact()
        self.update(a, r)
        return r
```

L'attributo ```bandit``` ha lasciato il posto a ```bandits``` che contiene la lista delle slot machine cui giocare, mentre gli attributi ```k``` e ```q``` sono diventati anch'essi delle liste contenenti per ogni slot machine il numero di volte che il giocatore ha giocato e la probabilità stimata di vincere. Si è aggiunto l'attributo ```n``` che contiene il numero di slot machine.

E' stato aggiunto il metodo ```choose()``` che in maniera casuale con una distribuzione uniforme sceglie la slot machine cui giocare.

Il metodo ```update(int, float)``` è stato modificato in modo da accettare come parametro di input l'indicazione della slot machine cui giocare, mentre il metodo ```do()``` in più invoca ```choose()```, in modo da scegliere la slot machine su cui giocare.

Facciamo un altro esperimento con tre slot machine e verifichiamo che la classe ```Agent``` riesca ad individuare quella con la probabilità di vincere più favorevole:

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
            r[i] = agent.do()

        rewards_trend: List[float] = np.cumsum(r) / np.arange(1, n + 1)
        plt.plot(rewards_trend, label='mean reward = {0:.5f}'.format(rewards_trend[-1]))

        for i in range(len(bandits)):
            plt.plot([0, n - 1], [bandits[i].pwin, bandits[i].pwin], label='pwin = {0:.5f}'.format(bandits[i].pwin))
            print('Bandit #{0} : q = {1:.5f}  pwin = {2:.5f}'.format(i, agent.q[i], bandits[i].pwin))

        plt.xscale('log')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    Experiment.main()
```

![](random_policy.png)

```
Bandit #0 : q = 0.00118  pwin = 0.00120
Bandit #1 : q = 0.00106  pwin = 0.00110
Bandit #2 : q = 0.00096  pwin = 0.00090
```
Cosa possiamo osservare?

1. La probabilità media stimata di vincere è 0.00107, esattamente uguale a quella teorica ossia, considerando che le giocate sono uniformemente distribuite fra le 3 slot machine è:

$$
\large \frac{0.00120 + 0.0011 + 0.0009}{3}=0.00107
$$

2. Siamo riusciti ad individuare la slot machine con la probabilità di vincere a noi più favorevole, ossia è quella che ha il valore di ```q``` più alto, ossia la prima.

3. Non abbiamo sfruttato quest'ultima informazione, in quanto abbiamo adoperato tutte le giocate e quindi tutte i gettoni a nostra disposizione, solo per individuare la slot machine migliore con la quale giocare, ma poi non ci abbiamo giocato.

Miglioriamo il nostro ```Agent``` e facciamo in modo che una parte delle giocate sia adoperata per individuare la slot machine migliore con cui giocare e la parte restante per giocare alla migliore:

```python
from typing import List
from base.Bandit import Bandit
import numpy as np

class Agent:
    def __init__(self, epsilon: float, bandits: List[Bandit]):
        self.epsilon: float = epsilon
        self.bandits: List[Bandit] = bandits
        self.n: int = len(bandits)
        self.k: List[int] = [0] * self.n
        self.q: List[float] = [0] * self.n

    def choose(self) -> int:
        a: int = 0
        p: float = np.random.random()
        if p < self.epsilon:
            a = np.random.choice(self.n)
        else:
            a = np.argmax(self.q)
        return a

    def update(self, a: int, r: float):
        self.q[a] += (r - self.q[a]) / (self.k[a] + 1)
        self.k[a] += 1

    def do(self) -> float:
        a: int = self.choose()
        r: float = self.bandits[a].interact()
        self.update(a, r)
        return r
```

Vediamo cos'è cambiato:

- è stato aggiunto l'attributo ```epsilon``` che indica la percentuale delle giocate da adoperare per la ricerca della slot machine migliore; se ```epsilon=0.05``` significa che il 5% delle giocate verrà adoperato a questo scopo;
- il metodo ```choose()``` genera un numero casuale uniformemente distribuito fa 0 ed 1, se questo è inferiore a ```epsilon``` la slot machine viene scelta a caso come prima, altrimenti si sceglie quella che fino a quel momento ha la probabilità stimata di vincere più alta.

Facciamo un esperimento:

```python
from typing import List
from base.Bandit import Bandit
from epsilongreedy.Agent import Agent
import numpy as np
import matplotlib.pyplot as plt

class Experiment:
    @staticmethod
    def main():
        reels: int = 3
        symbols: int = 10
        deltas: List[float] = [0.0002, 0.0001, -0.0001]
        epsilon: float = 0.05
        n: int = 1000000

        bandits: List[Bandit] = [Bandit(reels, symbols, delta) for delta in deltas]
        agent: Agent = Agent(epsilon, bandits)

        r: List[int] = [0] * n
        for i in range(n):
            r[i] = agent.do()

        rewards_trend: List[float] = np.cumsum(r) / np.arange(1, n + 1)
        plt.plot(rewards_trend, label='mean reward = {0:.5f}'.format(rewards_trend[-1]))

        for i in range(len(bandits)):
            plt.plot([0, n - 1], [bandits[i].pwin, bandits[i].pwin], label='pwin = {0:.5f}'.format(bandits[i].pwin))
            print('Bandit #{0} : q = {1:.5f}  pwin = {2:.5f}'.format(i, agent.q[i], bandits[i].pwin))

        plt.xscale('log')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    Experiment.main()
```
![](epsilon_greedy_policy.png)

```
Bandit #0 : q = 0.00120  pwin = 0.00120
Bandit #1 : q = 0.00101  pwin = 0.00110
Bandit #2 : q = 0.00110  pwin = 0.00090
```

La cosa che balza all'occhio è che adesso la probabilità che abbiamo di vincere è salita a 0.00119 e, considerando che la massima teorica è 0.0012, possiamo ritenerci più che soddisfatti.

## Cosa abbiamo imparato?

Bene... traiamo le nostre conclusioni.

~~~sequence
```mermaid
sequenceDiagram
    participant Alice
    participant Bob
    Alice->John: Hello John, how are you?
    loop Healthcheck
        John->John: Fight against hypochondria
    end
    Note right of John: Rational thoughts <br/>prevail...
    John-->Alice: Great!
    John->Bob: How about you?
    Bob-->John: Jolly good!
```
~~~



Per cominciare nell'apprendimento con rinforzo ...

PARLARE DEL PROBLEMA DELL'EXPLORATION-EXPLOITATION!!!

### APPUNTI  ADDESTRAMENTO  CON  RINFORZO

Serve per permettere ad un <u>agente</u> di agire in un <u>ambiente</u>.

Quando l'agente compie delle azioni nell'ambiente, riceve dei <u>feedback</u> (<u>ricompense</u>) dall'ambiente. Il feedback può essere anche negativo, in questo caso piuttosto che di ricompensa si dovrebbe parlare di <u>penalità</u>, però se consideriamo la penalità come una <u>ricompensa negativa</u>, possiamo continuare ad usare il termine ricompensa.

L'agente deve raggiungere un <u>obiettivo</u> (<u>goal</u>).

Le diverse <u>configurazioni</u> nelle quali può trovarsi l'ambiente costituiscono gli <u>stati</u> dell'ambiente.

L'obiettivo dell'agente è <u>massimizzare le ricompense future</u> e non solo quelle immediate (<i>spiegare meglio</i>).

<u>Azione</u>: E' ciò che l'agente può fare nell'ambiente.

**L'ambiente di trova in uno stato s(t), l'agente compie un'azione a(t), questa fa passare l'ambiente nello stato s(t+1) e l'agente riceve una ricompensa (che può essere negativa) r(t+1).**

