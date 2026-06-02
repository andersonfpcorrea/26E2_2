## 1. Configuração e reprodutibilidade

Esta seção fixa o ambiente do pipeline. O projeto é gerenciado com **uv**
(Python 3.12) e todas as dependências estão travadas em `uv.lock`. Os dados do
NLTK ficam em um diretório local ao projeto (`nltk_data/`) e o modelo do spaCy
(`en_core_web_sm`) é declarado como dependência, eliminando passos manuais de
download. Fixamos `random_state = 42` em todas as etapas estocásticas
(divisão treino/teste, t-SNE, Word2Vec, modelagem de tópicos) para garantir
resultados determinísticos.

A célula a seguir carrega o corpus já caracterizado (`data/processed.parquet`),
define os diretórios de saída e imprime as versões das principais bibliotecas.
