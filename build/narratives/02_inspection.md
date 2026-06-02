## 2. Carga e inspeção do corpus

O corpus **BBC News** (D. Greene e P. Cunningham, ICML 2006 — fonte:
http://mlg.ucd.ie/datasets/bbc.html) reúne **2.225 notícias** publicadas pela
BBC em 2004-2005, distribuídas em **5 editorias**: _business_, _entertainment_,
_politics_, _sport_ e _tech_. As classes são razoavelmente equilibradas (de 386
a 511 documentos por categoria), o que dispensa reamostragem e simplifica a
avaliação dos classificadores.

**Por que este corpus?** Ele satisfaz simultaneamente todos os critérios
exigidos: tem volume suficiente (> 1.000 documentos), textos longos (média de
~384 palavras por documento, bem acima do mínimo recomendado de 200), rótulos
claros para classificação, e é densamente povoado por **entidades nomeadas**
(pessoas, organizações, lugares) — matéria-prima ideal para NER e para o grafo
de conhecimento. Além disso, é público, citável e roda em CPU, sem GPU.

A inspeção inicial confirma a distribuição de classes e o comprimento dos
documentos por categoria.
