from varstore import GLITCHES

def is_recog_glitch(spoken, glitches=GLITCHES):
    for g in glitches:
        if spoken.find(g) != -1:
            return True
    return False