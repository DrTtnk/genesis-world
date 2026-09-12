import xml.etree.ElementTree as ET

import pytest


@pytest.fixture
def hinge_bone_xml():
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    world = ET.SubElement(root, "worldbody")
    base = ET.SubElement(world, "body", name="base")
    ET.SubElement(base, "geom", type="box", size=".01 .02 .01", pos="-.06 0 0", mass=".05")
    bone = ET.SubElement(base, "body", name="bone", pos="0 0 0")
    ET.SubElement(bone, "joint", name="hinge", axis="0 1 0", range="-.6 .6", damping=".002", armature="0")
    ET.SubElement(bone, "geom", type="box", size=".05 .01 .005", pos=".05 0 0", mass=".1")
    return ET.tostring(root, encoding="unicode")
