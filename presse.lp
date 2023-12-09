\* Minimizzazione_Cambi_Stampo *\
Minimize
OBJ: assignment_('A49',_1) + assignment_('A49',_2) + assignment_('A49',_3)
 + assignment_('A50',_1) + assignment_('A50',_2) + assignment_('A50',_3)
 + assignment_('A54',_1) + assignment_('A54',_2) + assignment_('A54',_3)
 + assignment_('A82',_1) + assignment_('A82',_2) + assignment_('A82',_3)
Subject To
_C1: assignment_('A54',_1) + assignment_('A54',_2) + assignment_('A54',_3)
 >= 53.66
_C10: assignment_('A54',_3) = 0
_C11: assignment_('A50',_1) = 0
_C12: assignment_('A50',_2) = 0
_C13: assignment_('A50',_3) = 0
_C14: assignment_('A49',_1) = 0
_C15: assignment_('A49',_2) = 0
_C16: assignment_('A49',_3) = 0
_C17: assignment_('A82',_1) = 0
_C18: assignment_('A82',_2) = 0
_C19: assignment_('A82',_3) = 0
_C2: assignment_('A50',_1) + assignment_('A50',_2) + assignment_('A50',_3)
 >= 0.6
_C3: assignment_('A49',_1) + assignment_('A49',_2) + assignment_('A49',_3)
 >= 11.7
_C4: assignment_('A82',_1) + assignment_('A82',_2) + assignment_('A82',_3)
 >= 36.48
_C5: assignment_('A54',_1) = 0
_C6: assignment_('A54',_2) = 0
_C7: assignment_('A54',_3) = 0
_C8: assignment_('A54',_1) = 0
_C9: assignment_('A54',_2) = 0
Binaries
assignment_('A49',_1)
assignment_('A49',_2)
assignment_('A49',_3)
assignment_('A50',_1)
assignment_('A50',_2)
assignment_('A50',_3)
assignment_('A54',_1)
assignment_('A54',_2)
assignment_('A54',_3)
assignment_('A82',_1)
assignment_('A82',_2)
assignment_('A82',_3)
End
