from amc_parser import parse_amc, parse_asf
from viewer import Viewer

if __name__ == '__main__':
    asf_path = '../data/p2s1.asf'
    amc_path = '../data/p2s1.amc'
    joints = parse_asf(asf_path)
    motions = parse_amc(amc_path)
    frame_idx = 0
    joints['root'].set_motion(motions[frame_idx])
    joints['root'].draw()

    joints = parse_asf(asf_path)
    motions = parse_amc(amc_path)
    v = Viewer(joints, motions)
    v.run()
