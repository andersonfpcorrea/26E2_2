## 6. Reconhecimento de entidades e grafo de conhecimento

Aplicamos o reconhecedor de entidades nomeadas (NER) do spaCy
(`en_core_web_sm`) sobre o **texto bruto** dos 2.225 documentos da BBC, usando
`nlp.pipe` com `batch_size=64` e desativando _parser_, _tagger_ e _lemmatizer_
para ganhar velocidade — mantendo `tok2vec` e `ner`, que são os componentes
necessários ao reconhecimento. O resultado mostra um corpus jornalístico típico:
predominam **PERSON** e **ORG**, seguidos de **DATE** e **GPE**. As entidades
mais frequentes — _US_, _UK_, _England_, _Labour_, _Britain_ — revelam de imediato
o eixo geopolítico britânico e internacional que organiza a cobertura.

Em paralelo ao NER estatístico, aplicamos **extração por expressões regulares**
para campos estruturados que o NER nem sempre normaliza bem: valores monetários
(ex.: `$1.13bn`, `£600m`), percentuais (ex.: `76%`, `2%`), datas (ex.: `Friday`),
além de e-mails e URLs. Isso confirma a forte presença de números financeiros e
estatísticos no jornalismo econômico e esportivo da BBC.

Antes de construir o grafo, fizemos uma etapa de **normalização difusa (fuzzy)**
com a biblioteca `rapidfuzz`. As formas de superfície foram primeiro limpas de
pronomes de tratamento e sufixos corporativos (_Mr_, _Inc_, _Corp_ etc.) e depois
agrupadas por similaridade `token_sort_ratio >= 88`, unindo variantes como
"Tony Blair"/"Mr Blair" ou "Microsoft"/"Microsoft Corp". Essa etapa reduz
milhares de formas de superfície a um conjunto canônico, diminuindo o ruído da
análise de rede.

O **grafo de conhecimento** (NetworkX) usa como nós as entidades canônicas mais
frequentes (tipos PERSON, ORG, GPE, NORP, LOC, FAC, EVENT) e como arestas a
**coocorrência no mesmo documento**, ponderada pela contagem e filtrando arestas
fracas (peso < 3). Calculamos duas centralidades: **grau** (conectores mais
densos) e **intermediação/betweenness** (pontes entre comunidades).

A leitura combinada é reveladora. Pela centralidade de grau dominam entidades
nacionais — _UK_, _Europe_, _US_, _London_ —, mostrando que a cobertura gira em
torno de um núcleo britânico de alcance internacional. Já a centralidade de
intermediação destaca pontes temáticas: _Liverpool_, _Oscar_, _Google_ e
_Hollywood_ ligam clusters distintos (esporte, entretenimento, tecnologia). Ou
seja: os hubs nacionais conectam tudo de forma densa, enquanto entidades de nicho
funcionam como elos entre os diferentes domínios noticiosos. Visualizamos o grafo
de forma estática (matplotlib) e interativa (PyVis), além de exportá-lo em
GraphML para análise posterior.

**Pergunta analítica respondida pelo grafo:** _Quais entidades são os principais
conectores na cobertura da BBC e o que isso revela?_ A resposta: os conectores
mais centrais por grau são entidades geopolíticas (UK, US, Europe, London),
confirmando uma agenda noticiosa centrada no Reino Unido e de enquadramento
internacional; já a centralidade de intermediação aponta entidades de nicho
(Liverpool, Oscar, Google) como pontes que ligam comunidades temáticas distintas.
