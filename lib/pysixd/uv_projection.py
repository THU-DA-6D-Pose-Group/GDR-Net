import numpy as np
import sys
from scipy import spatial
import os.path as osp

# pip install ai.cs
from ai import cs  # coordinates transformation

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../.."))

from lib.pysixd.inout import load_ply_model, load_ply


def get_uv_vertex_kdtrees(models, to_255=True):
    """build the map: vertex uv --> vertex 3d coordinate."""
    kdtrees = []
    for model in models:
        uv = model["texture_uv"]

        points = model["pts"]
        # uv_points_dict = {(uu, vv): points[i] for i, (uu, vv) in enumerate(uv)}

        # Calculate distances to the nearest neighbors from pts_gt to pts_est
        # nn_index = spatial.cKDTree(pts_est)
        # nn_dists, _ = nn_index.query(pts_gt, k=1)
        if to_255:
            uv_255 = (uv * 255 + 0.5).astype(np.uint8)
            uv_index_knn = spatial.cKDTree(uv_255)
        else:
            uv_index_knn = spatial.cKDTree(uv)
        # uv_dists, uv_indices = uv_index_knn.query(uv, k=1)
        # points_from_uv_index_knn = points[uv_indices]
        kdtrees.append((uv_index_knn, points))
    return kdtrees


"""if uv are calculated, we can store the uv to coordinate map?

    or we can calculate the mapping online using this transformer.
    assume z is the up direction
    method 1: calculate uv by coordinates with projection function (done)
            TODO: consider r as the 3rd channel(0~255) in order to handle models like can with genus>0
                If consider r, it just transform the cartesian coordinate to a polar coordinate,
                however the output space is still under R^3 or 256^3
    method 2(TODO): find the nearest projected points on the model surface and assign the uv,
            need to find the intersaction point between the spherical or cylindrical vector
            and the model surface (using ray-triangle intersection)
            maybe this method is used in DPOD paper to make the mapping is bijection
    TODO: conformal mapping, don't know how and whether it works??
"""


class CoordinatesTransformer(object):
    def __init__(self, cls_name, model_path_or_model, projection_type=None, to_255=False):
        """
        projection_type: 'cylindrical'/'cylinder', 'spherical'/'sphere'
        """
        self.cls_name = cls_name
        # model_dir = 'data/LINEMOD_6D/models'
        # model_path = osp.join(model_dir, '{0}/{0}.ply'.format(cls_name))
        if isinstance(model_path_or_model, str):
            self.points_cart = load_ply_model(model_path_or_model)
        else:
            self.points_cart = model_path_or_model
        self.projection_type = projection_type
        self.uv = None
        self.r = None
        self.z = None  # for cylindrical projection
        # self.uv_points_dict = None
        # self.uv_index_knn = None  # if use float points as key, both kdtree and dict can be lossless
        if projection_type is not None:
            self.points_to_uv(self.points_cart, project_type=self.projection_type, to_255=to_255)

    @staticmethod
    def project_to_sphere(points):
        """project cartesian to spherical surface, resulted in the surface cartesian coordinates
        points: Nx3
        ----

        """
        # for uv, the sphere: r=1, azimuth(phi): 2*pi*u, elevation(theta): 2*pi*v
        # theta is elevation, phi is azimuth
        r, theta, phi = cs.cart2sp(x=points[:, 0], y=points[:, 1], z=points[:, 2])
        # logger.info(f"number of zero points in r: {np.sum(r==0)}")
        assert np.sum(r == 0) == 0, "points contains zeros"
        points_sphere = points / r.reshape(-1, 1)
        return points_sphere, r, theta, phi

        # r, theta, phi = cs.cart2sp(x=1, y=1, z=1)

        # # spherical to cartesian
        # x, y, z = cs.sp2cart(r=1, theta=np.pi/4, phi=np.pi/4)

        # # cartesian to cylindrical
        # r, phi, z = cs.cart2cyl(x=1, y=1, z=1)

    @staticmethod
    def project_to_cylinder(points):
        """project model points onto a unit cylinder, r=1, h=1.

        points: Nx3
        """
        r, phi, z = cs.cart2cyl(points[:, 0], points[:, 1], points[:, 2])
        # NOTE: z is unchanged
        # print(np.array_equal(z, points[:, 2]))
        # project onto unit cylinder
        x_cyl = points[:, 0] / r
        y_cyl = points[:, 1] / r
        z_cyl = (z - z.min()) / (z.max() - z.min())
        points_cyl = np.stack((x_cyl, y_cyl, z_cyl), axis=1)
        return points_cyl, r, phi, z

    @staticmethod
    def spherical_to_cartesian(points_sp):
        pass

    @staticmethod
    def cylindrical_to_cartesian(points_cyl):
        pass

    @staticmethod
    def spherical_to_uv(points_sp):
        x, y, z = points_sp[:, 0], points_sp[:, 1], points_sp[:, 2]
        u = (np.arctan2(y, x) % (2 * np.pi)) / (2 * np.pi)  # [0, 1]
        v = (np.arcsin(z) % (2 * np.pi)) / (2 * np.pi)
        uv = np.stack((u, v), axis=1)
        return uv

    @staticmethod
    def cylindrical_to_uv(points_cyl):
        x, y, z = points_cyl[:, 0], points_cyl[:, 1], points_cyl[:, 2]
        u = (np.arctan2(y, x) % (2 * np.pi)) / (2 * np.pi)
        v = z
        uv = np.stack((u, v), axis=1)
        return uv

    @staticmethod
    def uv_to_cylindrical(uv):
        pass

    @staticmethod
    def uv_to_spherical(uv):
        pass

    def points_to_uv(self, points, project_type="cylinder", to_255=True):
        # print(f"{projection_type}")
        # print(f"{self.projection_type}")
        if project_type in ["cylindrical", "c", "cylinder"]:
            self.projection_type = "cylindrical"
            r, phi, z = cs.cart2cyl(x=points[:, 0], y=points[:, 1], z=points[:, 2])
            z_01 = (z - z.min()) / (z.max() - z.min())
            # u = phi/(2*pi), v = z_01
            u = (phi % (2 * np.pi)) / (2 * np.pi)
            v = z_01
            self.z = z
        elif project_type in ["s", "spherical", "sphere"]:
            self.projection_type = "spherical"
            r, theta, phi = cs.cart2sp(x=points[:, 0], y=points[:, 1], z=points[:, 2])
            # u == phi/(2*pi), v == theta/(2*pi)
            u = (phi % (2 * np.pi)) / (2 * np.pi)
            v = (theta % (2 * np.pi)) / (2 * np.pi)
        else:
            raise NotImplementedError("Wrong projection_type: {}".format(project_type))
        uv = np.stack((u, v), axis=1)
        if to_255:
            # NOTE: not bijection
            # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
            uv = (uv * 255 + 0.5).astype(np.uint8)
        self.uv_points_dict = {(uu, vv): points[i] for i, (uu, vv) in enumerate(uv)}

        # Calculate distances to the nearest neighbors from pts_gt to pts_est
        # nn_index = spatial.cKDTree(pts_est)
        # nn_dists, _ = nn_index.query(pts_gt, k=1)
        self.uv_index_knn = spatial.cKDTree(uv)
        self.uv = uv
        self.r = r
        return uv, r, self.z

    def uv_to_points(self, uv, r, projection_type="cylindrical", is_255=True, z=None):
        if is_255:
            uv = uv / 255.0
        u, v = uv[:, 0], uv[:, 1]
        if projection_type in ["cylindrical", "c"]:
            x_cyl = np.cos(2 * np.pi * u)
            y_cyl = np.sin(2 * np.pi * u)
            # z_01 = v
            if z is not None:
                z = z
            elif self.projection_type == "cylindrical" and self.z is not None:
                z = self.z
            else:
                raise ValueError("z should be provided")
            # from unit cylinder to model
            x = x_cyl * r
            y = y_cyl * r
            points = np.stack((x, y, z), axis=1)
        elif projection_type in ["s", "spherical"]:
            x = np.cos(2 * np.pi * v) * np.cos(2 * np.pi * u)
            y = np.cos(2 * np.pi * v) * np.sin(2 * np.pi * u)
            z = np.sin(2 * np.pi * v)
            points_sphere = np.stack((x, y, z), axis=1)  # Nx3
            points = points_sphere * r.reshape((-1, 1))  # from unit sphere to object
        else:
            raise NotImplementedError("Wrong projection_type: {}".format(projection_type))

        return points
