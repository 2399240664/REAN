
count = 0
correct_1step = 0
correct_2step = 0
# correct_3step = 0

with open ("data/all_as.txt","r") as f1, open ("data/QA/ans.txt","r") as f2:
    line1 = f1.readline()
    line2 = f2.readline()
    while line1 and line2:
        count=count+1
        a_step1 = line1.split("#")[1]
        a_step2 = line1.split("#")[3]
        # a_step3 = line1.split("#")[5]

        b_step1 = line2.split("#")[1]
        b_step2 = line2.split("#")[3]
        # b_step3 = line2.split("#")[5]

        if a_step1 == b_step1:
            correct_1step=correct_1step+1
        if a_step2 == b_step2:
            correct_2step=correct_2step+1
        # if a_step3 == b_step3:
        #     correct_3step = correct_3step + 1
        line1 = f1.readline()
        line2 = f2.readline()
print(count,correct_1step,correct_2step)

print(correct_1step/count,correct_2step/count)

