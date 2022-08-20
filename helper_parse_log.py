








log_file = './tmp2.log'


if __name__ == "__main__":

    with open(log_file,'r') as f:

        problem_ids = []
        good_ids = []

        cLine = 0
        cID = str()
        for l in f:
            s = l.split(' ')
            if l.startswith('Loaded paper'):
                if cLine == 0:
                    good_ids.append(cID)
                else:
                    problem_ids.append(cID)
                cLine = 0
                cID = s[2]
            elif l.startswith('Overwritting existing string for key') or l.startswith('Entry type online not standard'):
                continue
            else:
                cLine += 1

        print("PROBLEM IDS")
        print("-" * 20)
        for p_id in problem_ids:
            print(p_id.strip())

        print("GOOD IDS")
        print("-" * 20)
        for g_id in good_ids:
            print(g_id.strip())

