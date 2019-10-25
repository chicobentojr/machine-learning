# Ex. de programação 2: Q-learning e Sarsa


Vamos usar o framework de IA de Berkeley. Os exercícios de programação são em Python 2.7.

Roteiro:

1. Acessar o material de Reinforcement learning: http://ai.berkeley.edu/reinforcement.html
2. Baixar o kit de programação [NESTE LINK](https://moodle.inf.ufrgs.br/mod/resource/view.php?id=85478) (se preferir usar o original de Berkeley, substitua os arquivos `gridworld.py` e `sarsaAgents.py` pelos do nosso link. Caso contrário, o suporte ao agente Sarsa e o BookCliffGrid não estará disponível)
3. Fazer as **Questões 4 e 5**
4. Implementar o agente Sarsa: siga as mesmas instruções das **Questões 4** e **5**, com as seguintes adaptações:
   1. Não execute o Sarsa com controle manual e/ou aleatório (os valores ficarão errados porque a classe não foi projetada para atualizar os valores da política executada manualmente pelo usuário).
   2. Preencha os métodos no arquivo `sarsaAgents.py` ao invés de `qlearningAgents.py`. Observe que o agente Sarsa tem uma função `computeAction` que o Qlearning não tem. Pense (não precisa responder): por que ela é necessária?
   3. Substitua `-a q` por `-a s` nas chamadas a `gridworld.py`. Isso irá carregar o agente Sarsa ao invés do Qlearning. Por exemplo, para executar o agente Sarsa com 5 episódios de treinamento (demais parâmetros funcionam normalmente):
   
        ```
        python gridworld.py -a s -k 5 
        ```
    4. Não executar o `autograder.py` e o `crawler.py`. Eles não estão configurados para executar agentes Sarsa por enquanto. Ao invés disso, faça um experimento correspondente ao Exemplo 6.6 do livro . Para recompensa 'padrão' de -1, ambiente determinístico (sem ruido) e gamma = 1, a chamada é: `python gridworld.py -r -1 -n 0 -d 1 -g BookCliffGrid` (demais parâmetros especificados normalmente). Treine por 1000 episódios ou mais (dica: use o parâmetro -q para pular a exibição do aprendizado e acelerar a execução). Observe se as políticas resultantes do Sarsa e do Qlearning são iguais. Observe, também, qual agente tem melhor média de recompensa por episódio ao longo do treinamento (essa informação é exibida no terminal). Verifique para vários valores de epsilon (e.g. 0, 0.1, 0.3).
5. Fazer a **Questão 6**.
6. **Enviar 3 arquivos** neste link de envio (postar os arquivos diretamente, sem zipar):
   1. Agente Qlearning: `qlearningAgents.py`
   2. Agente Sarsa: `sarsaAgents.py`
   3. Questão 6: `analysis.py`, com o método `question6()` preenchido com sua resposta.