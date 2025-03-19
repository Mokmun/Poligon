import torch
import torch.nn as nn
import torch_geometric
from lightconvpoint import spatial

from lightconvpoint.nn import Convolution_FKAConv as Conv
from lightconvpoint.nn import max_pool, interpolate
from lightconvpoint.spatial import knn, sampling_quantized as sampling
from torch_geometric.data import Data

class ResidualBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,  neighborhood_size, ratio=None, n_support=None):
        super().__init__()

        self.cv0 = nn.Conv1d(in_channels, in_channels//2, 1)
        self.bn0 = nn.BatchNorm1d(in_channels//2)
        self.cv1 = Conv(in_channels//2, in_channels//2, kernel_size)
        self.bn1 = nn.BatchNorm1d(in_channels//2)
        self.cv2 = nn.Conv1d(in_channels//2, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)

        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.bn_shortcut = nn.BatchNorm1d(out_channels) if in_channels != out_channels else nn.Identity()
        

    
    def forward(self, x, pos, support_points, neighbors_indices):

        if x is not None:
            x_short = x
            x = self.activation(self.bn0(self.cv0(x)))
            x = self.activation(self.bn1(self.cv1(x, pos, support_points, neighbors_indices)))
            x = self.bn2(self.cv2(x))

            x_short = self.bn_shortcut(self.shortcut(x_short))
            if x_short.shape[2] != x.shape[2]:
                x_short = max_pool(x_short, neighbors_indices)

            x = self.activation(x + x_short)

        return x


class FKAConvNetwork(torch.nn.Module):

    def __init__(self, in_channels, out_channels, segmentation=False, hidden=64, dropout=0.5):
        super().__init__()

        self.lcp_preprocess = True
        self.segmentation = segmentation

        self.cv0 = Conv(in_channels, hidden, 16)
        self.bn0 = nn.BatchNorm1d(hidden)

        
        if self.segmentation:

            self.resnetb01 = ResidualBlock(hidden, hidden, 16, 16, ratio=1)
            self.resnetb10 = ResidualBlock(hidden, 2*hidden, 16, 16, ratio=0.25)
            self.resnetb11 = ResidualBlock(2*hidden, 2*hidden, 16, 16, ratio=1) 
            self.resnetb20 = ResidualBlock(2*hidden, 4*hidden, 16, 16, ratio=0.25)
            self.resnetb21 = ResidualBlock(4*hidden, 4*hidden, 16, 16, ratio=1)
            self.resnetb30 = ResidualBlock(4*hidden, 8*hidden, 16, 16, ratio=0.25)
            self.resnetb31 = ResidualBlock(8*hidden, 8*hidden, 16, 16, ratio=1)
            self.resnetb40 = ResidualBlock(8*hidden, 16*hidden, 16, 16, ratio=0.25)
            self.resnetb41 = ResidualBlock(16*hidden, 16*hidden, 16, 16, ratio=1)
            self.cv5 = nn.Conv1d(32*hidden, 16 * hidden, 1)
            self.bn5 = nn.BatchNorm1d(16*hidden)
            self.cv3d = nn.Conv1d(24*hidden, 8 * hidden, 1)
            self.bn3d = nn.BatchNorm1d(8 * hidden)
            self.cv2d = nn.Conv1d(12 * hidden, 4 * hidden, 1)
            self.bn2d = nn.BatchNorm1d(4 * hidden)
            self.cv1d = nn.Conv1d(6 * hidden, 2 * hidden, 1)
            self.bn1d = nn.BatchNorm1d(2 * hidden)
            self.cv0d = nn.Conv1d(3 * hidden, hidden, 1)
            self.bn0d = nn.BatchNorm1d(hidden)
            self.fcout = nn.Conv1d(hidden, out_channels, 1)
        else:
            
            self.resnetb01 = ResidualBlock(hidden, hidden, 16, 16, ratio=1)
            self.resnetb10 = ResidualBlock(hidden, 2*hidden, 16, 16, n_support=512)
            self.resnetb11 = ResidualBlock(2*hidden, 2*hidden, 16, 16, ratio=1) 
            self.resnetb20 = ResidualBlock(2*hidden, 4*hidden, 16, 16, ratio=0.25)
            self.resnetb21 = ResidualBlock(4*hidden, 4*hidden, 16, 16, ratio=1)
            self.resnetb30 = ResidualBlock(4*hidden, 8*hidden, 16, 16, ratio=0.25)
            self.resnetb31 = ResidualBlock(8*hidden, 8*hidden, 16, 16, ratio=1)
            self.resnetb40 = ResidualBlock(8*hidden, 16*hidden, 16, 16, ratio=0.25)
            self.resnetb41 = ResidualBlock(16*hidden, 16*hidden, 16, 16, ratio=1)
            self.fcout = nn.Linear(1024, out_channels)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, data, spatial_only=False, spectral_only=False, return_all_decoder_features=False):

        pos = data["pos"]

        squeeze_after_computation=False
        if len(pos.shape) == 2:
            pos = pos.unsqueeze(0)
            squeeze_after_computation = True
        pos = pos.transpose(1,2)


        if not spectral_only:
            # compute the support points
            support1, _ = sampling(pos, 0.25)
            support2, _ = sampling(support1, 0.25)
            support3, _ = sampling(support2, 0.25)
            support4, _ = sampling(support3, 0.25)

            # compute the ids
            ids00 = knn(pos, pos, 16)
            ids01 = knn(pos, support1, 16)
            ids11 = knn(support1, support1, 16)
            ids12 = knn(support1, support2, 16)
            ids22 = knn(support2, support2, 16)
            ids23 = knn(support2, support3, 16)
            ids33 = knn(support3, support3, 16)
            ids34 = knn(support3, support4, 16)
            ids44 = knn(support4, support4, 16)

            if squeeze_after_computation:
                support1 = support1.squeeze(0)
                support2 = support2.squeeze(0)
                support3 = support3.squeeze(0)
                support4 = support4.squeeze(0)

                ids00 = ids00.squeeze(0)
                ids01 = ids01.squeeze(0)
                ids11 = ids11.squeeze(0)
                ids12 = ids12.squeeze(0)
                ids22 = ids22.squeeze(0)
                ids23 = ids23.squeeze(0)
                ids33 = ids33.squeeze(0)
                ids34 = ids34.squeeze(0)
                ids44 = ids44.squeeze(0)
            
            data["support1"] = support1
            data["support2"] = support2
            data["support3"] = support3
            data["support4"] = support4

            data["ids00"] = ids00
            data["ids01"] = ids01
            data["ids11"] = ids11
            data["ids12"] = ids12
            data["ids22"] = ids22
            data["ids23"] = ids23
            data["ids33"] = ids33
            data["ids34"] = ids34
            data["ids44"] = ids44

            if self.segmentation:
                ids43 = knn(support4, support3, 1)
                ids32 = knn(support3, support2, 1)
                ids21 = knn(support2, support1, 1)
                ids10 = knn(support1, pos, 1)
                
                if squeeze_after_computation:
                    ids43 = ids43.squeeze(0)
                    ids32 = ids32.squeeze(0)
                    ids21 = ids21.squeeze(0)
                    ids10 = ids10.squeeze(0)
                
                data["ids43"] = ids43
                data["ids32"] = ids32
                data["ids21"] = ids21
                data["ids10"] = ids10


        if not spatial_only:
            x = data["x"].transpose(1,2)
            pos = data["pos"].transpose(1,2)

            x0 = self.activation(self.bn0(self.cv0(x, pos, pos, data["ids00"])))
            x0 = self.resnetb01(x0, pos, pos, data["ids00"])
            x1 = self.resnetb10(x0, pos, data["support1"], data["ids01"])
            x1 = self.resnetb11(x1, data["support1"], data["support1"], data["ids11"])
            x2 = self.resnetb20(x1, data["support1"], data["support2"], data["ids12"])
            x2 = self.resnetb21(x2, data["support2"], data["support2"], data["ids22"])
            x3 = self.resnetb30(x2, data["support2"], data["support3"], data["ids23"])
            x3 = self.resnetb31(x3, data["support3"], data["support3"], data["ids33"])
            x4 = self.resnetb40(x3, data["support3"], data["support4"], data["ids34"])
            x4 = self.resnetb41(x4, data["support4"], data["support4"], data["ids44"])

            if self.segmentation:
                
                x5 = x4.max(dim=2, keepdim=True)[0].expand_as(x4)
                x4d = self.activation(self.bn5(self.cv5(torch.cat([x4, x5], dim=1))))
                
                x3d = interpolate(x4d, data["ids43"])
                x3d = self.activation(self.bn3d(self.cv3d(torch.cat([x3d, x3], dim=1))))

                x2d = interpolate(x3d, data["ids32"])
                x2d = self.activation(self.bn2d(self.cv2d(torch.cat([x2d, x2], dim=1))))
                
                x1d = interpolate(x2d, data["ids21"])
                x1d = self.activation(self.bn1d(self.cv1d(torch.cat([x1d, x1], dim=1))))
                
                xout = interpolate(x1d, data["ids10"])
                xout = self.activation(self.bn0d(self.cv0d(torch.cat([xout, x0], dim=1))))
                xout = self.dropout(xout)
                xout = self.fcout(xout)

            else:

                xout = x4
                xout = xout.mean(dim=2)
                xout = self.dropout(xout)
                xout = self.fcout(xout)

            data["x"] = xout

        return data



        if x is not None:
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            x = x.transpose(1,2)


        print(pos.shape, support1.shape, support2.shape)
        exit()


        if self.segmentation:
            if ("net_indices" in data) and (data["net_indices"] is not None):
                ids0, ids10, ids11, ids20, ids21, ids30, ids31, ids40, ids41, ids3u, ids2u, ids1u, ids0u = data["net_indices"]
            else:
                ids0, ids10, ids11, ids20, ids21, ids30, ids31, ids40, ids41, ids3u, ids2u, ids1u, ids0u = [None for _ in range(13)]
        else:
            if ("net_indices" in data) and (data["net_indices"] is not None):
                ids0, ids10, ids11, ids20, ids21, ids30, ids31, ids40, ids41 = data["net_indices"]
            else:
                ids0, ids10, ids11, ids20, ids21, ids30, ids31, ids40, ids41 = [None for _ in range(9)]

        if ("net_support" in data) and (data["net_support"] is not None):
            support1, support2, support3, support4 = data["net_support"]
        else:
            support1, support2, support3, support4 = [None for _ in range(4)]


        ids0 = knn(pos, pos, 16, ids0)
        x0 = self.cv0(x, pos, pos, ids0)
        if x0 is not None:
            x0 = self.activation(self.bn0(x0))
        x0, _, _ = self.resnetb01(x0, pos, pos, ids0)

        x1, support1, ids10 = self.resnetb10(x0, pos, support1, ids10)
        x1, _, ids11 = self.resnetb11(x1, support1, support1, ids11)
        x2, support2, ids20 = self.resnetb20(x1, support1, support2, ids20)
        x2, _, ids21 = self.resnetb21(x2, support2, support2, ids21)
        x3, support3, ids30 = self.resnetb30(x2, support2, support3, ids30)
        x3, _, ids31 = self.resnetb31(x3, support3, support3, ids31)
        x4, support4, ids40 = self.resnetb40(x3, support3, support4, ids40)
        x4, _, ids41 = self.resnetb41(x4, support4, support4, ids41)
            
        if self.segmentation:
            xout = x4
            ids3u = knn(support4, support3, 1, ids3u)
            ids2u = knn(support3, support2, 1, ids2u)
            ids1u = knn(support2, support1, 1, ids1u)
            ids0u = knn(support1, pos, 1, ids0u)
            
            if xout is not None:
                x5 = xout.max(dim=2, keepdim=True)[0].expand_as(xout)
                x4d = self.activation(self.bn5(self.cv5(torch.cat([xout, x5], dim=1))))
                
                x3d = interpolate(x4d, ids3u)
                x3d = self.activation(self.bn3d(self.cv3d(torch.cat([x3d, x3], dim=1))))

                x2d = interpolate(x3d, ids2u)
                x2d = self.activation(self.bn2d(self.cv2d(torch.cat([x2d, x2], dim=1))))
                
                x1d = interpolate(x2d, ids1u)
                x1d = self.activation(self.bn1d(self.cv1d(torch.cat([x1d, x1], dim=1))))
                
                xout = interpolate(x1d, ids0u)
                xout = self.activation(self.bn0d(self.cv0d(torch.cat([xout, x0], dim=1))))
                xout = self.dropout(xout)
                xout = self.fcout(xout)
            
                if return_all_decoder_features:
                    xout = [x4d, x3d, x2d, x1d, xout]

            output_data = Data(outputs=xout,
                               net_support=[support1, support2, support3, support4], 
                               net_indices=[ids0, ids10, ids11, ids20, ids21, ids30, ids31, ids40, ids41, ids3u, ids2u, ids1u, ids0u]
                               )

            if xout is None:
                for support_id, support in enumerate(output_data["net_support"]):
                    output_data["net_support"][support_id] = support.squeeze(0)
                for ids_id, ids in enumerate(output_data["net_indices"]):
                    output_data["net_indices"][ids_id] = ids.squeeze(0)

            return output_data

        else:
            xout = x4
            if xout is not None:
                xout = xout.mean(dim=2)
                xout = self.dropout(xout)
                xout = self.fcout(xout)
            output_data = Data(outputs=xout, 
                                net_support=[support1, support2, support3, support4], 
                                net_indices=[ids0, ids10, ids11, ids20, ids21, ids30, ids31, ids40, ids41])

            if xout is None:
                for support_id, support in enumerate(output_data["net_support"]):
                    output_data["net_support"][support_id] = support.squeeze(0)
                for ids_id, ids in enumerate(output_data["net_indices"]):
                    output_data["net_indices"][ids_id] = ids.squeeze(0)
            return output_data