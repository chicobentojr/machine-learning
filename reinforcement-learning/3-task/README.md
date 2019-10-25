# Ex. de programação 3: Pacman & Elegibilidade

Vamos usar o framework de IA de Berkeley. Os exercícios de programação são em Python 2.

Para esta tarefa, comece com os agentes Q-learning e Sarsa desenvolvidos no Exercício 2 de programação. Porém, atualize o arquivo `pacman.py` com o [DESTE LINK](https://moodle.inf.ufrgs.br/pluginfile.php/142085/mod_resource/content/0/pacman.py), pois o original não tem suporte a Sarsa com traços de elegibilidade.

Roteiro:

1. PacMan: faça a **Questão 7** em http://ai.berkeley.edu/reinforcement.html. Nela, você avaliará sua implementação anterior de Q-learning no ambiente do PacMan.
2. Sarsa com traços de elegibilidade no **gridworld**:
   1. Implementar os traços do tipo 'accumulating traces', isto é, o traço do par estado-ação recém-visitado aumenta em 1
   2. Atenção: o parâmetro de decaimento dos traços de elegibilidade é definido na construtora de SarsaAgent com o nome `lamda` (sem o b) porque `lambda` é uma palavra chave do python.
   3. Treine o agente por 10 episódios, alpha=0.5, gamma=0.9, epsilon=0.1 e com os seguintes valores de lambda: 0, 0.1, 0.3, 0.5, 0.7, 0.9 e 1.0 no gridworld.py. O parâmetro lambda é especificado com --lambda na chamada ao `gridworld.py`.
   4. Faça dois gráficos de barras com a recompensa acumulada ao longo do treinamento, para cada valor de lambda. Cada barra deverá ter a média de 5 repetições, com barras de erro indicando o desvio padrão.
      1. Faça o primeiro gráfico rodando o gridworld.py com `-n 0.0` e `-r -0.04` para especificar que uma ação sempre tem o efeito desejado, e que a recompensa de todos os estados, exceto os terminais, é -0.04.
      2. Faça o segundo gráfico rodando o gridworld.py com `-n 0.2` e `-r -0.04` para especificar que uma ação tem 20% de chance de levar o agente a uma direção ortogonal à desejada, e que a recompensa de todos os estados, exceto os terminais, é -0.04.
3. Se você implementou o agente Sarsa e os traços de elegibilidade corretamente, ele também pode ser executado no PacMan.  Execute o agente Sarsa com traços de elegibilidade nesse ambiente (`python2 pacman.py -p PacmanSarsaAgent -x 2000 -n 2010 -l smallGrid -a lamda=[valor]`) e observe, para diferentes valores de lambda (usar o [`pacman.py`](https://moodle.inf.ufrgs.br/mod/resource/view.php?id=85532) fornecido neste exercício, pois o original não suporta traços de elegibilidade):
   1. Se ele acumula mais ou menos recompensa ao longo do treinamento
   2. Se ele vence todas as 10 partidas de teste
4. **Enviar 5 arquivos** neste link de envio (postar os arquivos diretamente, sem zipar):
   1. Agente Qlearning: `qlearningAgents.py`
   2. Agente Sarsa: `sarsaAgents.py`
   3. Gráfico de barra do Sarsa(lambda) no gridworld.py com `-n 0.0` e `-r -0.04`: `lambdas_deterministic.{pdf ou png}`
   4. Gráfico de barra do Sarsa(lambda) no gridworld.py com `-n 0.2` e `-r -0.04`: lambdas_stochastic.{pdf ou png}
   5. Suas observações do Sarsa(lambda) no ambiente do pacman: `sarsa_pacman.txt` (texto simples, sem formatação, com suas observações. Por exemplo, para qual valor de lambda o agente acumulou mais recompensa ao longo do treinamento, e se ele venceu todas as partidas no teste)