
from operator import concat
import numpy as np
from scipy.spatial import Delaunay
import scipy
import torch
def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag
def point_cloud_to_range_image(point_cloud, isMatrix,H=32,W=1024, semantic=None,rgb=None,\
                               return_semantic = False, return_remission = False, return_points=False,return_mask=False,moving_mask_name=None,return_scan = False):
    if (isMatrix):
        laser_scan = LaserScan(H=H,W=W,fov_up = 10.67, fov_down= -30.67,enable_cuda = False)
        laser_scan.set_points(point_cloud,remissions=None,semantic=semantic,rgb=rgb)
        laser_scan.do_range_projection()
    else:
        laser_scan = LaserScan(H=H,W=W,fov_up = 10.67, fov_down= -30.67,enable_cuda = False)
        
        laser_scan.open_scan(point_cloud,semantic = semantic,moving_mask_name=moving_mask_name)
    
        laser_scan.do_range_projection()
        
    
    return_results=[laser_scan.proj_range]

    if return_remission:
        return_results.append(laser_scan.proj_remission)
    if return_semantic:
        return_results.append(laser_scan.proj_semantic)
    if return_mask:
        return_results.append(laser_scan.proj_mask)
    if rgb is not None:
        return_results.append(laser_scan.proj_rgb)
    if return_points:
        # return laser_scan.proj_xyz, laser_scan.proj_range, laser_scan.proj_remission
        return_results.append(laser_scan.proj_xyz)
    if return_scan:
        return_results.append(laser_scan)

    return return_results
'''
    Class taken fom semantic-kitti-api project.  https://github.com/PRBonn/semantic-kitti-api/blob/master/auxiliary/laserscan.py
'''
class LaserScan:
    """Class that contains LaserScan with x,y,z,r"""
    EXTENSIONS_SCAN = ['.bin']

    def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0,enable_cuda = False):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        # enable cuda
        self.enable_cuda = enable_cuda
        self.device = 'cpu' if not enable_cuda else 'cuda'
        self.reset()

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros(
            (0, 3), dtype=np.float32)        # [m, 3]: x, y, z
        self.remissions = np.zeros(
            (0, 1), dtype=np.float32)    # [m ,1]: remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.float32)
        # projected semantic - [H,W] intensity (-1 is no data)
        self.proj_semantic = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.float32)
        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)       # [H,W] mask
        self.proj_rgb = np.zeros((self.proj_H, self.proj_W,3),
                                  dtype=np.float32)       # [H,W] mask
        
        if self.enable_cuda:
            self.points = torch.tensor(self.points).to(self.device)
            self.remissions = torch.tensor(self.remissions).to(self.device)
            self.proj_range = torch.tensor(self.proj_range).to(self.device)
            self.unproj_range = torch.tensor(self.unproj_range).to(self.device)
            self.proj_xyz = torch.tensor(self.proj_xyz).to(self.device)
            self.proj_remission = torch.tensor(self.proj_remission).to(self.device)
            self.proj_semantic = torch.tensor(self.proj_semantic).to(self.device).long()
            self.proj_idx = torch.tensor(self.proj_idx).to(self.device).long()
            self.proj_x = torch.tensor(self.proj_x).to(self.device)
            self.proj_y = torch.tensor(self.proj_y).to(self.device)
            self.proj_mask = torch.tensor(self.proj_mask).to(self.device)
            self.proj_rgb = torch.tensor(self.proj_rgb).to(self.device)
            # for data in data_list:
            #     data = torch.tensor(data).to(self.device) # load on gpu
            # import pdb;pdb.set_trace()
    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename,semantic=None,rgb=None,moving_mask_name=None):
        """ Open raw scan and fill in attributes
        """
        # reset just in case there was an open structure
        self.reset()

        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")

        # if all goes well, open pointcloud
        scan = np.fromfile(filename, dtype=np.float32)
        # scan = scan.reshape((-1, 4)) points in nuScenes: N*5
        scan = scan.reshape((-1, 5))

        # put in attribute
        points = scan[:, 0:3]    # get xyz
        dist = np.linalg.norm(points ,axis=-1)
        points = points[(dist>3)&(dist<80),:]
        remissions = scan[:, 3]  # get remission
        remissions =remissions[(dist>3)&(dist<80)]
        if moving_mask_name is not None:
            f = open(moving_mask_name,'r')
            bbox_file = f.readlines() 
            bboxes = np.array([box.split()[1:] for box in bbox_file]).astype(np.float32).reshape(-1,8,3) # get corners
            flag = np.zeros(points.shape[0],)
            for bbox in bboxes:

                flag += in_hull(points,bbox)
            points = points[flag==0]
            
        # print('original shape: {}',format(points.shape[0]))
        

        remissions = None
        self.set_points(points, remissions,semantic=semantic)

    def set_points(self, points, remissions=None,semantic=None,rgb=None):
        """ Set scan attributes (instead of opening from file)
        """
        # reset just in case there was an open structure
        self.reset()

        # check scan makes sense
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        # check remission makes sense
        if remissions is not None and not isinstance(remissions, np.ndarray):
            raise TypeError("Remissions should be numpy array")

        # put in attribute
        self.points = points    # get xyz
        if remissions is not None:
            self.remissions = remissions  # get remission
        else:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)
        if semantic is not None:
            self.semantic = semantic  # get remission
        else:
            self.semantic = np.zeros((points.shape[0]), dtype=np.float32)
        if rgb is not None:
            self.rgb = rgb
        else:
            self.rgb = np.zeros((points.shape[0],3), dtype=np.float32)
        # if projection is wanted, then do it and fill in the structure
        if self.enable_cuda:
            # for data in [self.points,self.rgb,self.semantic]:
            #     data = torch.tensor(data).to(self.device)
            self.points = torch.tensor(self.points).to(self.device)
            self.rgb = torch.tensor(self.rgb).to(self.device)
            self.semantic = torch.tensor(self.semantic).to(self.device)
            self.remissions = torch.tensor(self.remissions).to(self.device)
        if self.project:
            self.do_range_projection()

    def do_range_projection(self,drop_mask=None):
        """ Project a pointcloud into a spherical projection image.projection.
            Function takes no arguments because it can be also called externally
            if the value of the constructor was not set (in case you change your
            mind about wanting the projection)
        """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad
        if not self.enable_cuda:
            # get depth of all points
            depth = np.linalg.norm(self.points, 2, axis=1)

            # get scan components
            scan_x = self.points[:, 0]
            scan_y = self.points[:, 1]
            scan_z = self.points[:, 2]

            # get angles of all points
            yaw = -np.arctan2(scan_y, scan_x)
            pitch = np.arcsin(scan_z / depth)


            # get projections in image coords
            proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
            proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

            # scale to image size using angular resolution
            proj_x *= self.proj_W                              # in [0.0, W]
            proj_y *= self.proj_H                              # in [0.0, H]

            # round and clamp for use as index
            proj_x = np.floor(proj_x)
            proj_x = np.minimum(self.proj_W - 1, proj_x)
            proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
            self.proj_x = np.copy(proj_x)  # store a copy in orig order

            proj_y = np.floor(proj_y)
            proj_y = np.minimum(self.proj_H - 1, proj_y)
            proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
            self.proj_y = np.copy(proj_y)  # stope a copy in original order

            # copy of depth in original order
            self.unproj_range = np.copy(depth)

            # order in decreasing depth
            indices = np.arange(depth.shape[0])
            order = np.argsort(depth)[::-1]
            depth = depth[order]
            indices = indices[order]
            points = self.points[order]
            remission = self.remissions[order]
            semantic = self.semantic[order]
            rgb = self.rgb[order]
            proj_y = proj_y[order]
            proj_x = proj_x[order]

            # assing to images
            self.proj_range[proj_y, proj_x] = depth
            self.proj_xyz[proj_y, proj_x] = points
            self.proj_remission[proj_y, proj_x] = remission
            
            self.proj_semantic[proj_y, proj_x] = semantic
            self.proj_rgb[proj_y, proj_x] = rgb

            self.proj_idx[proj_y, proj_x] = indices
            self.proj_mask = (self.proj_idx > 0).astype(np.float32)
        else:
            # all the operatoration on the torch.tensor.cuda
             # get depth of all points
            depth = torch.linalg.norm(self.points, 2, dim=1)

            # get scan components
            scan_x = self.points[:, 0]
            scan_y = self.points[:, 1]
            scan_z = self.points[:, 2]

            # get angles of all points
            yaw = -torch.arctan2(scan_y, scan_x)
            pitch = torch.arcsin(scan_z / depth)


            # get projections in image coords
            proj_x = 0.5 * (yaw / torch.pi + 1.0)          # in [0.0, 1.0]
            proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

            # scale to image size using angular resolution
            proj_x *= self.proj_W                              # in [0.0, W]
            proj_y *= self.proj_H                              # in [0.0, H]

            # round and clamp for use as index
            proj_x = torch.floor(proj_x)

            proj_x = torch.minimum(torch.tensor(self.proj_W - 1).to(self.device).broadcast_to(proj_x.shape), proj_x)
            proj_x = torch.maximum(torch.tensor(0).to(self.device).broadcast_to(proj_x.shape), proj_x).type(torch.int32)   # in [0,W-1]
            self.proj_x = torch.clone(proj_x)  # store a copy in orig order

            proj_y = torch.floor(proj_y)
            proj_y = torch.minimum(torch.tensor(self.proj_H - 1).to(self.device).broadcast_to(proj_x.shape), proj_y)
            proj_y = torch.maximum(torch.tensor(0).to(self.device).broadcast_to(proj_x.shape), proj_y).type(torch.int32)   # in [0,H-1]
            self.proj_y = torch.clone(proj_y)  # stope a copy in original order

            # copy of depth in original order
            self.unproj_range = torch.clone(depth)

            # order in decreasing depth
            indices = torch.arange(depth.shape[0]).to(self.device)
            # import pdb;pdb.set_trace()
            order = torch.argsort(depth).flip(dims=[0])
            depth = depth[order]
            indices = indices[order]
            points = self.points[order]
            remission = self.remissions[order]
            semantic = self.semantic[order]
            rgb = self.rgb[order]
            proj_y = proj_y[order].round().long()
            proj_x = proj_x[order].round().long()

            # assing to images
            self.proj_range[proj_y, proj_x] = depth
            self.proj_xyz[proj_y, proj_x] = points
            self.proj_remission[proj_y, proj_x] = remission
            self.proj_semantic[proj_y, proj_x] = semantic
            self.proj_rgb[proj_y, proj_x] = rgb

            self.proj_idx[proj_y, proj_x] = indices
            self.proj_mask = (self.proj_idx > 0).type(torch.float32)
        # print('after shape: {}'.format((self.proj_mask==1).sum()))

        # if drop_mask is not None:

        
def real_to_var(real,size=1):
    H,W = real.shape
    # Array = np.zeros((H,W,9))
    # Array[...,0]=real
    # Array[...,1]= np.roll(real,1,axis=0)
    # Array[...,2]= np.roll(real,-1,axis=0)
    # Array[...,3]= np.roll(real,1,axis=1)
    # Array[...,4]= np.roll(real,-1,axis=1)
    # Array[...,5]= np.roll(np.roll(real,1,axis=0),1,axis = 0)
    # Array[...,6]= np.roll(np.roll(real,1,axis=0),-1,axis = 1)
    # Array[...,7]= np.roll(np.roll(real,-1,axis=0),1,axis = 1)
    # Array[...,8]= np.roll(np.roll(real,-1,axis=0),-1,axis = 1)
    # # var = np.mean(Array,axis = -1)
    Array = np.stack([np.roll(real,i,axis = 1) for i in range(-size,size)],axis=-1)
    var = np.var(Array,axis = -1)
    return var

if __name__ == '__main__':
    H=32;W=1024;
    
    import time
    t0= time.time()
    for j in range(100,110):
        for i in range(1000):
            point_cloud = np.load('/SSD_DISK/users/zhangjunge/DATA/allsimulation/dvgo_simulation_v14/{:04d}/points_{:04d}.npy'.format(j,i))
            semantic = np.load('/SSD_DISK/users/zhangjunge/DATA/allsimulation/dvgo_simulation_v14/{:04d}/points_semantic_{:04d}.npy'.format(j,i))
            rgb = np.load('/SSD_DISK/users/zhangjunge/DATA/allsimulation/dvgo_simulation_v14/{:04d}/points_rgb_{:04d}.npy'.format(j,i))
            laser_scan = LaserScan(H=H,W=W,fov_up = 10.67, fov_down= -30.67,enable_cuda = False)
            laser_scan.set_points(point_cloud,remissions=None,semantic=semantic,rgb=rgb)
            
            laser_scan.do_range_projection()
            import pdb;pdb.set_trace()
    t1= time.time()
    print(f'time consuming: {t1-t0}')