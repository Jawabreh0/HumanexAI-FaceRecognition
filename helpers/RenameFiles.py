import os

path = "/home/jawabreh/Desktop/Face-Recognition/evaluation/accuracy/accuracy-test-data/unknown/"

def rename_images(path):
    i = 149
    for filename in os.listdir(path):
        src = path + filename
        dst = path + str(i) + ".jpeg"
        os.rename(src, dst)
        i += 1

rename_images(path)
print("\n\n\tSUCCESS\n\n")