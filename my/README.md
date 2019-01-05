# Il bandito bracciuto
## Introduzione leggera ma non troppo all'apprendimento con rinforzo.

Siete stanchi di lavorare e volete passare una vita in vacanza?

Bene, andiamo al casinò, però non siamo degli sprovveduti, quindi dobbiamo scegliere con attenzione la nostra strategia: a cosa giochiamo?

Alla roulette?

Non sembra una buona idea, al tavolo c'è quel signore pelato che sta vincendo tutto!

Giochiamo a poker?

Ehi... ma non sappiamo giocare a poker!

Giochiamo alle slot machines!

Siamo andati in un casinò retrò, quindi le slot machine sono quelle meccaniche a 3 rulli con 20 simboli ciascuno e con una leva, il braccio, in inglese _arm_, che bisogna tirare per far ruotare i rulli. Si vince quando su tutti rulli compare lo stesso simbolo. Che probabilità abbiamo di vincere?

$$
\large p_{win}=\frac{1}{20^{3}}=0.000125
$$

Un po' bassa... ecco perché le chiamano macchinette mangia soldi ed i nostri amici inglesi le chiamano _bandit_... _armed bandit_ per la precisione, letteralmente _bandito bracciuto_!

Va bene, non disperiamoci, nel casinò c'è una file di slot machine, sono ancora quelle meccaniche e la probabilità di vincere non è certamente quella teorica che abbiamo calcolato.

Come facciamo ad individuare la slot machine con la probabilità di vincere a noi più favorevole?

Dobbiamo provare a giocare con tutte, registriamo per ogni slot machine _i_:

- il numero di volte che abbiamo vinto;
- il numero di volte che abbiamo giocato;

il rapporto ci fornisce una stima della probabilità di vincere:

$$
\large \widetilde{p}_{win_{i}}=\frac{n_{win_{i}}}{n_{i}}
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
- ```reels```: numero di rulli, nel nostro caso sarà 3;
- ```symbols```: numero di simboli presenti su ciascun rullo, nel nostro caso sarà 20;

e calcola la probabilità di vincere ```pwin```.

Il metodo ```pull``` simula l'azione di giocare alla slot machine, per cui genera un numero casuale uniformemente distribuito fra 0 ed 1: se il numero casuale è minore della proabilità di vincere, abbiamo vinto.

Abbiamo detto però, che le slot machine sono meccaniche, per cui la probabilità di vincere non è quella teorica, ma non può allontanarsi molto da questa, per cui aggiungiamo al costruttore un'ulteriore parametro ```delta``` che indica questo scostamento:

```python
import numpy as np

class Bandit:
    def __init__(self, reels: int, symbols: int, delta: float):
        self.pwin: float = 1 / np.power(symbols, reels) + delta

    def pull(self) -> bool:
        return np.random.rand() < self.pwin
```

Adesso dobbiamo modellare il giocatore, che gioca alle slot machine e registra le vincite per stimare la probabilità di vincita. Per fare questo, creiamo due classi, una modella il giocatore che registra le vincite per stimare la probabilità di vincere e l'altra modella l'ambiente che permette l'interazione fra il giocatore e la slot machine.

Cominciamo con il giocatore:

```python
class Agent:
    def __init__(self):
        self.bets = 0;
        self.winning_bets = 0;

    def update(self, winning_bet: bool) -> float:
        self.bets += 1
        if winning_bet:
            self.winning_bets += 1
        return self.winning_bets / self.bets
```

(SPIEGARE LA CLASSE)

Passiamo all'ambiente:
```python

```
