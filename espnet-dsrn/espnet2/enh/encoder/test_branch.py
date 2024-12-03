import torch

from collections import OrderedDict
from espnet2.enh.encoder.branch import Branch_Block
from torch.nn import ModuleList


output = torch.rand([1,3,257])
ilens = torch.tensor(3)


#def gen_branch_block(nums):
    #blocks = OrderedDict()
    #for i in range(nums):
        #blocks["branch"+str(i)] = Branch_Block()
nums = 2
blocks = torch.nn.ModuleList(Branch_Block() for _ in range(nums))

    #return blocks
    


#blocks = gen_branch_block(2)

print(blocks)


print(type(blocks))
#for block in blocks.values():
    #output, ilens = block(output, ilens)

for block in blocks:
    output, ilens = blocks(input, ilens)


#output, ilens = blocks(input, ilens)




