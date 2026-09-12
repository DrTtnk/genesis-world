import xml.etree.ElementTree as ET

import pytest


@pytest.fixture
def muscle_hinge_chain_xml():
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    world = ET.SubElement(root, "worldbody")
    base = ET.SubElement(world, "body", name="base")
    ET.SubElement(base, "geom", type="box", size=".01 .02 .02", pos="-.10 0 .03", mass=".1")
    upper = ET.SubElement(base, "body", name="upper", pos=".12 0 0")
    ET.SubElement(upper, "joint", name="upper_hinge", axis="0 1 0", range="-.05 .05", damping=".01")
    ET.SubElement(upper, "geom", type="box", size=".02 .04 .04", mass=".128")
    lower = ET.SubElement(upper, "body", name="lower", pos=".10 0 0")
    ET.SubElement(lower, "joint", name="lower_hinge", axis="0 1 0", range="-.05 .05", damping=".01")
    ET.SubElement(lower, "geom", type="box", size=".02 .04 .04", mass=".128")
    return ET.tostring(root, encoding="unicode")
