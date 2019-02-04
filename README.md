# Seminararbeit Demo

Thema: Künstliche Intelligenz zum Erraten von Passwörtern

In diesem Repository befinden sich die relevanten Teile des Quellcodes sowie einige Scripts, Inputs und Outputs. Der Sinn dieser Software ist nicht die praktische Anwendung. Stattdessen geht es um eine Demonstration dessen, was in meiner Seminararbeit bereits beschrieben wurde. Dennoch mussten einige Sachen geändert werden, um eine gewisse Nutzbarkeit außerhalb meines eigenen Rechners zu erreichen.

Die unten beschriebenen Befehle wurde bereits in diesem Repository ausgeführt, deshalb liegen bereits einige der entstehenden Dateien herum.

## Dependencies

- python3
- mxnet
- numpy
- pandas
- seaborn
- matplotlib

## Beispieleingabe

Um die Software zu demonstrieren, habe ich mir [geleakte Passwörter](https://hashes.org/leaks.php?id=942) heruntergeladen und mit einem Script (program/fp2/filepatcher.py) auf Strings der Länge 10 gefiltert. (data/gamigo_10_11) Das ist notwendig, um das GRU trainieren zu können.

## Markow-Kette

Zur Demonstration dieses Programms (program/mc3) sollte in der Wurzel des Repositories folgender Befehl ausgeführt werden:
```
$ python3 program/mc3 program/mc3/mc_demoscript
```

Dabei wird zunächst die Datei data/gamingo_10_11 mit einem n=3 analysiert, im Anschluss wird daraus eine Markow-Kette erstellt und zuletzt werden 10 Passwörter mit der Länge 10 basierend auf dieser Kette generiert.

## Feedforward-Netz

Um ein Feedforward-Netz zu trainieren kann folgender Befehl genutzt werden:
```
$ python3 program/nn1/fnn_trainer.py <anzahl versteckter schichten> <lernrate>
```

Somit wird die dazugehörige Datei mit den passenden Parametern erstellt. Daraufhin können zufällige Worte der Länge 10 mit folgendem Befehl generiert werden:

```
$ python3 program/nn1/ffn_loader.py <dateien (.params)> <anzahl versteckter schichten>
```

## GRU

Um ein GRU zu trainieren kann folgender Befehl genutzt werden:
```
$ python3 program/nn1/gru_trainer.py <anzahl versteckter schichten> <lernrate>
```

Analog zum FFN können auch hier zufällige Wörter generiert werden.

```
$ python3 program/nn1/gru_loader.py <dateien (.params)> <anzahl versteckter schichten>
```

Leider ist der Output nicht akkurat. Anders als in der Arbeit beschrieben kann ich mittlerweile mit Sicherheit sagen, dass diese Passwörter nicht der Verteilung eines fehlerfrei trainierten GRU entsprechen. Die Wahrscheinlichkeiten, die vom Netz zurückgegeben werden, um die Wörter zu generieren verändern sich nicht, wenn sich das vorherige Wort und dadurch der Hidden State ändert. Ich weiß nicht, ob das Problem im Netz selbst liegt (d.h. dass es falsch trainiert wurde), oder, ob der Code, um den Hidden State anzupassen, nicht funktioniert. Es ist natürlich weiterhin möglich, dass ein völlig anderer Bug vorliegt.

## Weitere bekannte Probleme

1. Wenn man neuronale Netze mit diesem Code trainiert, gibt es hinundwieder Fehlermeldungen aus der `multiprocessing`-Bibliothek, die mit MXNet zusammenhängen. Diese können jedoch einfach ignoriert werden.
2. Wenn man eine Markow-Kette generiert führen asynchrone Ausgaben dazu, dass sich die Statusmeldungen überlappen und vermischen.
3. Wenn man mehrere neuronale Netze hintereinander trainiert, kann es zu Fehlern kommen, die mit Tkinter zusammenhängen. Ich weiß nicht, wie sich das Problem beheben lässt. Ein Workaround ist, mit dem letzten begonnenen Training neu anzufangen.


