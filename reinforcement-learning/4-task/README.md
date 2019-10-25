# Ex. de programação 4: Dyna-Q

Vamos usar o framework de IA de Berkeley. Os exercícios de programação são em Python 2.

Para esta tarefa, use o kit atualizado, fornecido [NESTE LINK](https://moodle.inf.ufrgs.br/mod/resource/view.php?id=85709), ou pelo menos atualize os arquivos `gridworld.py` e  `dynaQAgents.py` com os do kit, pois o original não tem suporte a agentes Dyna-Q.

Roteiro:

1. Preencha os métodos em `dynaQAgents.py` para implementar o agente Dyna-Q.
   1. Observe que o agente recebe o número de passos/iterações de planejamento (planning) na construtora através do parâmetro plan_steps. O parâmetro kappa pode ser ignorado por enquanto.
   2. O arquivo tem um comentário recomendando adicionar o código do planejamento (planning) na função update. Pense (não precisa responder): em que outro lugar o planejamento poderia ocorrer? Fique à vontade para escolher onde colocar o planejamento.
2. Teste no **gridworld**:
   1. Compare o Q-learning e o Dyna-Q no mundo 4x3 determinístico (o Dyna-Q é chamado com: python2 gridworld.py -a d --plan-steps X (X é o número de passos de planejamento). Demais parâmetros são especificados normalmente. Gere um gráfico de barras com a média da recompensa acumulada de 5 execuções com 25 episódios de treinamento cada, para o Q-learning e para o Dyna-Q. Ambos devem ser executados com epsilon = 0.1, gamma=0.9 e alpha=0.5. O Dyna-Q deve ser executado com 5 passos de planejamento.
   2. Gere um gráfico de linhas para o Dyna-Q no discount grid (-g DiscountGrid na chamada do gridworld.py) com a recompensa acumulada pelo número de passos de planejamento. Cada ponto do gráfico deve ter a média da recompensa acumulada em 5 execuções com 50 episódios de treinamento cada. Teste de 5 a 50 passos de planejamento, aumentando de 5 em 5. Observe se o agente consegue atingir a recompensa de +10, mais distante no mapa. Execute com alpha=0.5, gamma=0.9, epsilon=0.1
   3. [OPCIONAL] Estenda o seu agente para implementar o bônus de exploração do Dyna-Q+. Para isso, use o parâmetro kappa recebido na construtora (ele é passado com --kappa X na chamada do gridworld.py). Gere o gráfico anterior com 3 linhas, uma para kappa=0 (Dyna-Q padrão), kappa = 0.1 e kappa=0.5 . Observe qual versão acumula mais recompensa ao longo do treinamento. Qual(is) descobre(m) a recompensa mais distante? Execute com alpha=0.5, gamma=0.9, epsilon=0.1. (Haverá uma alteração quanto ao Dyna-Q+ apresentado no livro: como nosso agente não recebe o conjunto de estados, não é possível inicializar o modelo para todos os pares estado-ação. Porém, no método getAction, o agente pode consultar as ações disponíveis em um estado e pode inicializar o modelo mesmo para ações não realizadas naquele estado. O modelo é inicializado "incorretamente" de propósito como se aquela ação retornasse ao estado atual com recompensa zero.)
3. **Enviar 3 arquivos** neste link de envio (postar os arquivos diretamente, sem zipar):
   1. Agente Dyna-Q: `dynaQAgents.py`
   2. Gráfico de barra do Dyna-Q vs Q-learning no mundo 4x3: `dyna-vs-ql.{pdf ou png}`
   3. Gráfico de linhas com o desempenho do Dyna-Q (e Dyna-Q+, se implementado) no discount grid: `dynas-discount-grid.{pdf ou png}`
4. [Para sua diversão] Seu agente Dyna-Q também funciona no Pacman se implementado corretamente. Basta executar com: python2 [pacman.py](https://moodle.inf.ufrgs.br/mod/resource/view.php?id=85532) -p PacmanDynaQAgent [demais parâmetros passados normalmente].