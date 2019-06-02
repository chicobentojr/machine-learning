k_ = 10
dataset = wdbc
folder = datasets/$(dataset)/t$(t_)_k$(k_)

brew-path: 
	test -d ~/.linuxbrew && eval $(~/.linuxbrew/bin/brew shellenv)
	test -d /home/linuxbrew/.linuxbrew && eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)
	test -r ~/.bash_profile && echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.bash_profile
	echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.profile

pipenv: 
	#cd machine-learning
	pipenv install
	pipenv shell

run: 
	#cd random-forest
	python decision_tree.py create datasets/benchmark.csv -s ";" -img=tree.png #-v debug

tocsv: 
	python script.py datasets/$(dataset)/$(dataset).data "Class" datasets/$(dataset)/$(dataset).csv

forest: 
	python random_forest.py create datasets/$(dataset)/$(dataset).data -s "," -t $(t_) -k $(k_) -img=datasets/$(dataset)/t$(t_)_a$(a_)_k$(k_)

wdbc_set: 
	python random_forest.py create datasets/$(dataset).csv -s "," -t $(t_) -k $(k_) #-json=$(folder) -img=$(folder)

	
