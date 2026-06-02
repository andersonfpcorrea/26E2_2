## 4. Representação vetorial e busca textual

Para transformar os textos em vetores numéricos exploramos três abordagens
complementares. A primeira, **Bag-of-Words** (CountVectorizer sobre o
`clean_text`), gerou um vocabulário de **21.958 termos**. Ela apenas conta
ocorrências, de modo que palavras genéricas e muito frequentes dominam o ranking,
sem distinguir o que é realmente discriminativo.

A segunda, **TF-IDF com n-gramas (1,2)**, pondera cada termo pela sua raridade no
corpus e incorpora bigramas, elevando o vocabulário para **76.012 termos**. Os
termos de maior peso médio já refletem melhor os temas do corpus (_film_, _game_,
_win_, _government_, _election_, _labour_), pois o componente IDF penaliza
palavras onipresentes e valoriza vocabulário característico de cada categoria.

Escolhemos o **TF-IDF como base do motor de busca** justamente por essa
propriedade: ele realça os termos distintivos e, combinado com a **similaridade
do cosseno**, mede a proximidade temática entre a consulta e cada documento de
forma robusta ao tamanho do texto. A consulta é projetada no mesmo espaço
vetorial ajustado e ranqueada contra todos os documentos. Os resultados
confirmam a qualidade da abordagem: a busca por _"stock market interest rates and
economy"_ retorna exclusivamente notícias de **business**; _"new smartphone
mobile technology"_ recupera apenas matérias de **tech** sobre celulares; e
_"football championship final match"_ traz majoritariamente conteúdo de
**sport**. As pontuações relativamente baixas (em torno de 0,2-0,3) são esperadas
em consultas curtas sobre documentos longos, mas a ordenação por relevância é
coerente.

A terceira abordagem, **Word2Vec** (vector_size=100, window=5, min_count=5),
aprende embeddings densos a partir do contexto, capturando relações semânticas.
Os vizinhos mais próximos são plausíveis: _film_ aproxima-se de _award_, _star_ e
_actor_; _government_ de _security_ e _plan_.

Por fim, a visualização **t-SNE** (TF-IDF reduzido via TruncatedSVD para 50
dimensões e depois projetado em 2D) mostra **cinco agrupamentos nítidos por
categoria**. _Sport_ e _entertainment_ ficam bem isolados, enquanto _business_ e
_politics_ aparecem adjacentes, com pequena sobreposição decorrente do
vocabulário econômico e de políticas públicas que ambos compartilham. Isso
evidencia que as representações vetoriais capturam bem a estrutura temática do
corpus.
