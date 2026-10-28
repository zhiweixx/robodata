# Third-party code and artifacts

RoboData's Apache-2.0 license covers original project contributions. It does not
change the licenses of dependencies, checkpoints, simulator images or datasets.
Operators obtain those artifacts themselves through the pinned setup process.

| Component | Included here | License / source record |
|---|---|---|
| ASPIRE runtime | Modified Python runtime and small configuration; no real-robot meshes | [LICENSE](../third_party/aspire/LICENSE), [NOTICE](../third_party/aspire/NOTICE), [source manifest](../third_party/aspire/SOURCE_MANIFEST.json) |
| RoboDojo | Compatibility patch and small utility package initializer | [MIT license](../deploy/notices/robodojo-LICENSE), pin in [sources.lock.json](../deploy/sources.lock.json) |
| XPolicyLab | Compatibility patch | [Apache-2.0 license](../deploy/notices/xpolicylab-LICENSE), pinned source |
| Isaac Lab | Compatibility patch | [BSD-3-Clause license](../deploy/notices/isaaclab-LICENSE), pinned source |
| Native CuRobo fork | Download recipe, no model or binaries | [Apache-2.0 license](../deploy/notices/curobo-LICENSE), pinned source |
| Contact-GraspNet PyTorch port | Download recipe, runtime repair helper and narrow regression fixture | [upstream license](../deploy/notices/graspnet-LICENSE.txt); model/source use retains its own terms |
| SAM3 | Download recipe and API integration; weights are not included | Gated upstream access/license; pinned source and weight hashes in the source lock |
| Isaac Sim 6 | Container preparation recipe; image is not included | NVIDIA Isaac Sim license must be accepted by the installing operator |
| Qwen and RoboDojo assets | Download recipes and metadata only | Their upstream model/dataset licenses apply |

The two ASPIRE patches are retained under `patches/aspire` and duplicated as package
resources where runtime provenance requires them. The CGN singleton repair checks
the pinned source and creates an in-process subclass; it does not edit downloaded
vendor code or alter confidence thresholds/ranking.

No stored access token, API key, model checkpoint, simulator image, asset bundle,
compiled kernel cache or experiment video is required to clone this repository.
Do not assume access to a gated model transfers with this source release.
