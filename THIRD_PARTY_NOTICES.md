# Third-party provenance and licensing notes

This source package contains the team's project code and install metadata.
Third-party implementations, K1 meshes/XML, dependency environments, caches,
and trained weights are not included in the intended source archive. Installation
fetches dependencies independently using the immutable references below.

| Dependency/source | Pinned origin |
|---|---|
| Course's minimized `booster_mjlab` package and K1 assets | `parasollab/cs498-booster-track`, commit `7c51db18821560cef4908964984488da7c946930`, subdirectory `course/hw3/vendor/booster_mjlab` |
| mjlab | `mujocolab/mjlab`, commit `b517e0c489139e7fcee95702cfb2b01931264985` |
| PyTorch | Official `https://download.pytorch.org/whl/cu130` or `/cpu` index, version 2.11.0 |
| Other dependencies | Exact sources, versions, and distribution hashes in root `uv.lock` |

## Booster snapshot

The course's `course/hw3/PROVENANCE.md` identifies the original Booster upstream
as `https://github.com/IntelligentRoboticsLab/booster_mjlab.git` at commit
`d92e11395d961096d639446c0a8a176f12bb51da`. Its author metadata names Harold
Ruiter. The course snapshot narrows registrations to the K1/Adam PPO subset
and records its original and retained files in the accompanying vendor manifests.

That provenance file explicitly reports **no top-level software LICENSE/COPYING
in the original snapshot**. No software or asset license is inferred here.
The included historical Bulma CSS license concerns Bulma only and does not
license the Booster implementation or K1 robot assets. This project does not
relicense them. Obtain the appropriate upstream/course permission before
redistributing those dependency files or a combined environment/asset archive.

Source links:

- [Course provenance at the pinned revision](https://github.com/parasollab/cs498-booster-track/blob/7c51db18821560cef4908964984488da7c946930/course/hw3/PROVENANCE.md)
- [Original upstream repository](https://github.com/IntelligentRoboticsLab/booster_mjlab)

## mjlab and installed packages

mjlab declares Apache-2.0 at the pinned revision. Preserve its license and notices
when redistributing it; this package instead fetches it as a dependency.
[Pinned mjlab metadata](https://github.com/mujocolab/mjlab/blob/b517e0c489139e7fcee95702cfb2b01931264985/pyproject.toml).

PyTorch, MuJoCo, MuJoCo-Warp, Warp, rsl-rl, and the other downloaded packages
remain subject to their own licenses and notices distributed with their wheels
or source. The dependency lock is a reproducibility record, not a license grant.
No broader permission is asserted by this file.
