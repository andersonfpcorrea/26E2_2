## 5. Modelagem: classificação supervisionada e modelagem de tópicos

Esta etapa cobre **duas frentes** do requisito 3: classificação supervisionada e
modelagem de tópicos. Optamos por **não** usar análise de sentimento porque a
prosa jornalística é majoritariamente neutra — o sentimento lexical (VADER)
agregaria ruído, não sinal.

Para a **classificação** representamos cada documento com **TF-IDF** sobre o
texto já normalizado e lematizado (`clean_text`), usando unigramas e bigramas
(`ngram_range=(1,2)`, até 20000 termos). O TF-IDF é a escolha natural para texto
jornalístico: pondera termos discriminativos e penaliza palavras ubíquas, e os
bigramas capturam expressões (_prime minister_, _world cup_) que separam bem as
cinco editorias. Comparamos três classificadores clássicos e complementares:
**MultinomialNB** (baseline probabilístico rápido e forte em contagens de
texto), **LinearSVC** (margem máxima, excelente em espaços esparsos de alta
dimensão) e **LogisticRegression** com `class_weight="balanced"`. O corpus tem
leve desbalanceamento (_sport_ 511 vs _entertainment_ 386); o
`class_weight="balanced"` reescala a perda pelo inverso da frequência de cada
classe, evitando que o modelo favoreça as categorias maiores. A divisão foi
estratificada (`test_size=0.25`, `random_state=42`) para preservar essa
distribuição em treino e teste.

Os três modelos ficaram acima de 98% de acurácia, o que mostra que as editorias
da BBC são linguisticamente muito separáveis. O **LinearSVC venceu** (F1-macro
0.9874), errando apenas 6 dos 557 documentos de teste. A matriz de confusão
revela que _entertainment_, _sport_ e _tech_ foram classificados sem erros, e que
a única confusão recorrente é **politics ↔ business** — coerente, pois ambas
compartilham vocabulário de economia, governo e mercado. O SVM linear se destaca
porque maximiza a margem entre classes em espaços esparsos, exatamente o cenário
do TF-IDF.

Na **modelagem de tópicos** ajustamos **LDA** (sobre contagens) e **NMF** (sobre
TF-IDF), ambos com 5 tópicos para espelhar as 5 classes conhecidas. O NMF
recuperou de forma quase perfeita as cinco editorias, com tópicos limpos para
esporte, política, tecnologia, entretenimento e negócios. O LDA chegou a
estrutura semelhante, mas misturou esporte e entretenimento em alguns tópicos. A
lição é que, mesmo **sem rótulos**, a estrutura temática latente do corpus
reproduz a taxonomia editorial — confirmando que as categorias supervisionadas
correspondem a agrupamentos linguísticos reais, e não a divisões arbitrárias.
