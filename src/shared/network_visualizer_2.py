
from nnv import NNV

layersList = [
    {"title":"input\n(relu)", "units": 3, "color": "darkBlue"},
    {"title":"hidden 1\n(relu)", "units": 3},
    {"title":"hidden 2\n(relu)", "units": 3, "edges_color":"red", "edges_width":2},
    {"title":"output\n(sigmoid)", "units": 1,"color": "darkBlue"},
]

NNV(layersList).render(save_to_file="tmp/neural_network-2.png")

