## 3. Pré-processamento textual (NLTK + spaCy)

O pré-processamento converte texto bruto em tokens limpos e comparáveis. O
pipeline aplica, em ordem: **normalização** (minúsculas, remoção de pontuação,
dígitos e símbolos), **tokenização** de palavras e de sentenças (NLTK),
**remoção de stopwords** (as 198 stopwords do inglês do NLTK acrescidas de 23
stopwords customizadas de domínio jornalístico, como _said_, _mr_, _would_,
_also_, que sobrevivem à lista genérica mas não carregam sinal temático) e,
por fim, **lematização**.

### Stemming vs. lematização

Comparamos as duas estratégias de redução morfológica sobre uma **base de tokens
idêntica**, para isolar o efeito da operação. O **stemming de Porter** é
agressivo: corta sufixos sem consultar um dicionário, colapsando o vocabulário
em ~32%, mas gerando radicais que não são palavras reais (_studies_ → _studi_,
_policies_ → _polici_, _companies_ → _compani_). A **lematização** (WordNet) é
conservadora (~11% de redução) e preserva palavras válidas (_policies_ →
_policy_).

**Decisão de projeto:** a coluna canônica `clean_text`, usada em toda a etapa
seguinte (busca, tópicos, rótulos do grafo), foi gerada com **lematização do
spaCy**, não com stemming. O motivo é direto: radicais mutilados prejudicariam a
legibilidade dos termos de busca, das palavras-chave dos tópicos e dos rótulos
do grafo de conhecimento. O stemming permanece como demonstração comparativa,
não como caminho de produção. (Demonstramos dois lematizadores — WordNet na
comparação e spaCy na produção — evidenciando o domínio das ferramentas.)

A análise de **POS tagging** com spaCy e o impacto do pré-processamento no
vocabulário são apresentados a seguir, junto da nuvem de palavras, do histograma
de comprimento e da distribuição de classes gramaticais.
