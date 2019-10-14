import subprocess

if __name__ == '__main__':

    plan_steps = map(str, range(5, 55, 5))
    kappas = map(str, [0, 0.1, 0.5])
    n_times = 5

    for kappa in kappas:
        print "Results for Kappa="+kappa+"\n"

        for step in plan_steps:

            strout = ""

            for i in range(0, n_times):
                if i == 0:
                    strout += "{:>2} Steps".format(step)

                command = ["python", "gridworld.py",
                           "--agent=d",
                           "--grid=DiscountGrid",
                           "--episodes=50",
                           "--noise=0.0",
                           "--epsilon=0.1",
                           "--discount=0.9",
                           "--learningRate=0.5",
                           "--kappa="+kappa,
                           "--plan-steps="+step,
                           "--quiet",
                           "--text"]

                output = subprocess.check_output(command)

                # Removes all text before the avarage return value
                output = output[output.find(':')+2:]
                # Removes all text after the avarage return value
                output = output[:output.find('\n')]
                # Replaces the dot by comma to fit in the sheet with BR pattern
                output = output.replace('.', ',').replace(' ', '')

                strout += "\t{:<15}".format(output[:15])

            print strout

        print "\n"

    exit()
