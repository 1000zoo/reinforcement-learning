def solution(dirs):
    a = [0,0]
    course = []

    for i in dirs:
        b = a.copy()
        temp = [a,b]
        if i == 'U':
            b[1] += 1
        elif i == 'D':
            b[1] -= 1
        elif i == 'R':
            b[0] += 1
        elif i == 'L':
            b[0] -= 1

        if b[0] > 5:
            b[0] -= 1
        elif b[0] < -5:
            b[0] += 1
        elif b[1] > 5:
            b[1] -= 1
        elif b[1] < -5:
            b[1] += 1
        else:
            reverse = sorted(temp, reverse = True)
            if temp not in course and reverse not in course:
                course.append(temp)
        a = b.copy()

    return len(course)

print(solution("UDU"))