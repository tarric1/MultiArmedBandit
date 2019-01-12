# Il bandito bracciuto
## Introduzione leggera ma non troppo all'apprendimento con rinforzo.

Siete stanchi di lavorare e volete passare una vita in vacanza?

Bene, andiamo al casinò, però non siamo degli sprovveduti, quindi dobbiamo scegliere con attenzione la nostra strategia: a cosa giochiamo?

Alla roulette?

Non sembra una buona idea, al tavolo c'è quel signore pelato che sta vincendo tutto!

Giochiamo a poker?

Ehi... ma non sappiamo giocare a poker!

Giochiamo alle slot machine!

Siamo andati in un casinò retrò, quindi le slot machine sono quelle meccaniche a 3 rulli con 20 simboli ciascuno e con una leva, il braccio, in inglese _arm_, che bisogna tirare per far ruotare i rulli. Si vince quando su tutti rulli compare lo stesso simbolo. Che probabilità abbiamo di vincere?

$$
\large p_{win}=\frac{1}{20^{3}}=0.000125
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

Facciamo una simulazione: adoperiamo Python!

> Sono un fanatico dell'object oriented, per cui adopererò il supporto all'OOP che fortunatamente Python fornisce e, lo so che molti mi prenderanno per pazzo, ma non voglio far arrabbiare Zio Bob (Robert C. Martin, se non lo conoscete fatevi aiutare da Google e scoprirete perché), ed adopererò lo static typing che Python supporta decentemente dalla versione 3.6.

Tanto per cominciare dobbiamo modellare il bandito, la slot machine:

```python
import numpy as np

class Bandit:
    def __init__(self, reels: int, symbols: int):
        self.pwin: float = 1 / np.power(symbols, reels)

    def pull(self) -> bool:
        return np.random.rand() < self.pwin
```

Vediamo com'è fatta la classe ```Bandit```.

Il costruttore accetta come parametri di input:
- ```reels```: numero di rulli;
- ```symbols```: numero di simboli presenti su ciascun rullo;

e calcola la probabilità di vincere ```pwin```.

Il metodo ```pull``` simula l'azione di giocare alla slot machine, per cui genera un numero casuale uniformemente distribuito fra 0 ed 1, se è minore della probabilità di vincere, abbiamo vinto.

Abbiamo detto però, che le slot machine sono meccaniche, per cui la probabilità di vincere non è quella teorica, ma non può allontanarsi molto da questa, per cui aggiungiamo al costruttore un'ulteriore parametro ```delta``` che indica questo scostamento:

```python
import numpy as np

class Bandit:
    def __init__(self, reels: int, symbols: int, delta: float):
        self.pwin: float = 1 / np.power(symbols, reels) + delta

    def pull(self) -> bool:
        return np.random.rand() < self.pwin
```

Adesso dobbiamo modellare il giocatore, che gioca alle slot machine e registra le vincite per stimare la probabilità di vincita:

```python
class Agent:
    def __init__(self):
        self.bets: int = 0
        self.winning_bets: int = 0

    def update(self, winning_bet: bool):
        self.bets += 1
        if winning_bet:
            self.winning_bets += 1

    def estimate(self) -> float:
        return float(self.winning_bets) / self.bets
```

La classe ```Agent``` ha due attributi:

- ```bets```: numero di volte che il giocatore ha giocato;
- ```winning_bets```: numero di volte che il giocatore ha vinto;

corrispondono rispettivamente a:
$$
\large n_{i}, n_{win_{i}}
$$
La classe inoltre ha il metodo ```update``` che simula l'azione di registrazione del numero di giocate (```bets```) e del numero di vincite (```winning_bets```) e il metodo ```estimate``` che calcola la probabilità stimata di vincere.

Manca una componente che modella l'interazione fra il giocatore e la slot machine; questo lavoro è lasciato all'ambiente nel quale si trovano i due attori; rappresenta il mezzo che permette il contatto fra loro:

```python
from Bandit import Bandit
from Agent import Agent

class Environment:
    def __init__(self, bandit: Bandit, agent: Agent):
        self.bandit = bandit
        self.agent = agent

    def doInteraction(self):
        winning_bet: bool = self.bandit.pull()
        self.agent.update(winning_bet)
```

La classe ```Environment``` ha due attributi:

- ```bandit```: la slot machine;
- ```agent```: il giocatore;

e il metodo ```doInteraction``` che simula l'interazione fra la slot machine e il giocatore: esegue sulla prima l'azione di tirare il braccio, acquisisce l'esito della giocata e comunica l'esito al secondo.

Infine eseguiamo l'esperimento:

```python
from Bandit import Bandit
from Agent import Agent
from Environment import Environment

class Experiment:
    @staticmethod
    def main():
        reels: int = 3
        symbols: int = 20
        delta: float = 0
        n: int = 100000

        bandit: Bandit = Bandit(reels, symbols, delta)
        agent: Agent = Agent()
        environment: Environment = Environment(bandit, agent)

        for _ in range(n):
            environment.doInteraction()

        est_pwin: float = agent.estimate()
        print('Estimated pwin: %f' % est_pwin)

if __name__ == "__main__":
    Experiment.main()
```

La classe ```Experiment``` ha il metodo statico ```main``` che crea l'oggetto rappresentate la slot machine, quello rappresentante il giocatore, quindi crea l'ambiente ed esegue 100000 interazioni e per concludere visualizza la probabilità stimata di vincere.

Segue un esempio d'esecuzione dell'esperimento:

```
Estimated pwin: 0.000100
```

Abbiamo adesso le basi per andare avanti: ricordiamo che il 







### APPUNTI  ADDESTRAMENTO  CON  RINFORZO

Serve per permettere ad un <u>agente</u> di agire in un <u>ambiente</u>.

Quando l'agente compie delle azioni nell'ambiente, riceve dei <u>feedback</u> (<u>ricompense</u>) dall'ambiente. Il feedback può essere anche negativo, in questo caso piuttosto che di ricompensa si dovrebbe parlare di <u>penalità</u>, però se consideriamo la penalità come una <u>ricompensa negativa</u>, possiamo continuare ad usare il termine ricompensa.

L'agente deve raggiungere un <u>obiettivo</u> (<u>goal</u>).

Le diverse <u>configurazioni</u> nelle quali può trovarsi l'ambiente costituiscono gli <u>stati</u> dell'ambiente.

L'obiettivo dell'agente è <u>massimizzare le ricompense future</u> e non solo quelle immediate (<i>spiegare meglio</i>).

<u>Azione</u>: E' ciò che l'agente può fare nell'ambiente.

**L'ambiente di trova in uno stato s(t), l'agente compie un'azione a(t), questa fa passare l'ambiente nello stato s(t+1) e l'agente riceve una ricompensa (che può essere negativa) r(t+1).**

