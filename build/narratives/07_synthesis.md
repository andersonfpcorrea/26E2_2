## 7. Síntese e comunicação dos resultados

### Síntese em linguagem não técnica

Imagine uma redação com **2.225 reportagens** misturadas e a tarefa de organizá-las,
encontrá-las e entendê-las automaticamente. Foi isso que este pipeline fez, em
quatro movimentos:

1. **Organizar a linguagem.** Primeiro, "limpamos" os textos — removendo palavras
   sem valor informativo e reduzindo cada palavra à sua forma essencial — para que
   o computador comparasse ideias, não ruído.

2. **Encontrar por significado.** Construímos um buscador que, a partir de uma
   frase como _"mercado financeiro e juros"_, devolve as reportagens mais
   relevantes — e ele acerta o tema: pergunta sobre economia traz notícias de
   economia; pergunta sobre celulares traz notícias de tecnologia.

3. **Classificar com altíssima precisão.** Um modelo aprende a adivinhar a
   editoria de uma notícia (esporte, política, economia, tecnologia,
   entretenimento) e acerta **mais de 98%** das vezes. O único engano recorrente
   acontece entre _política_ e _economia_ — justamente porque, na vida real, esses
   assuntos se misturam (governo, impostos, mercado). Surpreendentemente, quando
   pedimos ao computador para descobrir os temas **sem nenhuma pista**, ele
   reencontra praticamente as mesmas cinco editorias: prova de que essas
   categorias refletem divisões reais da linguagem, não rótulos arbitrários.

4. **Mapear quem é quem.** Extraímos automaticamente as pessoas, empresas e
   lugares citados e desenhamos um "mapa de relações". Esse mapa mostra que a
   cobertura gira em torno de um eixo britânico e internacional (Reino Unido, EUA,
   Europa), enquanto nomes mais específicos (Liverpool, Oscar, Google) funcionam
   como "pontes" que conectam mundos diferentes — esporte, cinema e tecnologia.

**Conclusão para o decisor:** é possível, com ferramentas abertas e sem
infraestrutura cara, transformar uma pilha de textos em um sistema que busca,
classifica e mapeia informação de forma confiável e interpretável. O mesmo
pipeline se aplica, com ajustes mínimos, a chamados de suporte, avaliações de
clientes, reclamações ou documentos internos.

### Principais achados técnicos

- As cinco editorias são **linguisticamente muito separáveis** (F1-macro ≈ 0,99).
- Métodos supervisionados e não supervisionados **convergem** para a mesma
  estrutura de cinco temas — uma triangulação que valida os rótulos.
- A confusão _política ↔ economia_ é o único ponto sistemático e é
  **semanticamente justificável**.
- O grafo de conhecimento expõe a **geografia editorial** da BBC: hubs nacionais
  densos e entidades de nicho como pontes entre domínios.

### Limitações e melhorias futuras

O corpus vem de uma única fonte (BBC, 2004-2005), então os modelos não
generalizam para outros veículos ou épocas sem reavaliação. O grafo baseia-se em
coocorrência (heurística), não em relações verificadas; um próximo passo seria
extrair triplas sujeito-verbo-objeto com o parser de dependências para relações
explícitas. Outras melhorias: embeddings contextuais (BERT) para busca semântica
mais fina, validação cruzada estratificada com intervalos de confiança, e um NER
customizado para entidades de domínio específico.
