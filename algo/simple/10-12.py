def minimumWaitingTime(queries):
    queries = sorted(queries)
    total_waiting_time = 0
    for idx, duration in enumerate(queries):
        count_left = len(queries) - idx - 1
        total_waiting_time += count_left * duration
    return total_waiting_time


def classPhotos(redShirtHeights, blueShirtHeights):
    # Write your code here.
    redShirtHeights = sorted(redShirtHeights)
    blueShirtHeights = sorted(blueShirtHeights)

    if redShirtHeights[0] == blueShirtHeights[0]:
        return False
    firstRow = redShirtHeights if redShirtHeights[0] < blueShirtHeights[0] else blueShirtHeights
    secondRow = redShirtHeights if redShirtHeights[0] > blueShirtHeights[0] else blueShirtHeights

    for idx in range(len(firstRow)):
        if firstRow[idx] >= secondRow[idx]:
            return False

    return True


def tandemBicycle(redShirtSpeeds, blueShirtSpeeds, fastest):
    # Write your code here.
    if fastest:
        redShirtSpeeds = sorted(redShirtSpeeds)
        blueShirtSpeeds = sorted(blueShirtSpeeds, reverse=True)
    else:
        redShirtSpeeds = sorted(redShirtSpeeds)
        blueShirtSpeeds = sorted(blueShirtSpeeds)
    totalSpeed = 0
    for idx, r in enumerate(redShirtSpeeds):
        totalSpeed += max(r, blueShirtSpeeds[idx])

    return totalSpeed


if __name__ == '__main__':
    a = [3, 2, 1, 2, 6]
