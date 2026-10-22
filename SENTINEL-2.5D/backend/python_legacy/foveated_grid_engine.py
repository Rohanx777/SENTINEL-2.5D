"""Distance-adaptive 2.5D aggregation. Geometry heuristics are not safety validation."""
import json
import math
import numpy as np
from coordinate_transformer import OdometryPose,CoordinateTransformer,make_polar_annular_wedge
FOVEATED_ZONES=[
    dict(zone_id=0,name='Near field',r_min=0.5,r_max=10.,delta_r=0.5,num_sectors=128,offset=0,purpose='Detailed nearby geometry'),
    dict(zone_id=1,name='Mid field',r_min=10.,r_max=30.,delta_r=1.,num_sectors=64,offset=2432,purpose='Obstacle representation'),
    dict(zone_id=2,name='Far field',r_min=30.,r_max=100.,delta_r=2.5,num_sectors=32,offset=3712,purpose='Coarse scene context'),
]

class FoveatedGridEngine:
    def __init__(self,history_len=1):
        self.zones=FOVEATED_ZONES
        self.history_len=history_len

    def project_point_cloud(self,points,labels=None,pose=None,compact=False):
        points=np.asarray(points)
        if labels is None: labels=np.zeros(len(points),dtype=np.uint8)
        if len(labels)!=len(points): raise ValueError('Point and label counts differ')
        minima=np.full(4608,np.inf);maxima=np.full(4608,-np.inf);sums=np.zeros(4608)
        counts=np.zeros(4608,dtype=np.int32);votes=np.zeros((4608,3),dtype=np.int32)
        radius=np.hypot(points[:,0],points[:,1]);angle=np.arctan2(points[:,1],points[:,0])
        for zone in self.zones:
            mask=(radius>=zone['r_min'])&(radius<zone['r_max'])&(points[:,2]>-4)&(points[:,2]<12)
            ring=np.floor((radius[mask]-zone['r_min'])/zone['delta_r']).astype(int)
            sector=np.minimum(zone['num_sectors']-1,np.floor((angle[mask]+np.pi)*zone['num_sectors']/(2*np.pi)).astype(int))
            cell=zone['offset']+ring*zone['num_sectors']+sector
            z=points[mask,2]
            np.minimum.at(minima,cell,z);np.maximum.at(maxima,cell,z);np.add.at(sums,cell,z);np.add.at(counts,cell,1)
            label=np.asarray(labels)[mask]
            for k in range(3): np.add.at(votes[:,k],cell,(label==k).astype(int))
        return self.project_aggregates((minima,maxima,sums,counts,votes),len(points),pose,compact)

    def project_aggregates(self,aggregates,total_points,pose=None,compact=False):
        minima,maxima,sums,counts,votes=aggregates
        pose=pose or OdometryPose()
        cells=[];zones=[]
        for zone in self.zones:
            size=int((zone['r_max']-zone['r_min'])/zone['delta_r'])*zone['num_sectors']
            zone_slice=slice(zone['offset'],zone['offset']+size)
            ids=np.flatnonzero(counts[zone_slice]>=2)+zone['offset']
            zone_cells=[]
            for index in ids:
                ring,sector=divmod(int(index)-zone['offset'],zone['num_sectors'])
                r0=zone['r_min']+ring*zone['delta_r'];r1=r0+zone['delta_r']
                p0=-math.pi+sector*2*math.pi/zone['num_sectors'];p1=p0+2*math.pi/zone['num_sectors']
                radius=(r0+r1)/2;phi=(p0+p1)/2;cx=radius*math.cos(phi);cy=radius*math.sin(phi)
                zmin=float(minima[index]);zmax=float(maxima[index]);span=zmax-zmin
                vote=int(np.argmax(votes[index]))
                label=('road','static_obstacle','dynamic_target')[vote]
                # Threshold-based geometry annotations; never claimed as learnt detection.
                if zone['zone_id']<=1 and zmin < -1.82 and span<0.6: label='negative_obstacle'
                elif zone['zone_id']==0 and 0.08<=span<=0.28 and vote==0: label='curb'
                if compact:
                    label_id={'road':0,'static_obstacle':1,'dynamic_target':2,'negative_obstacle':3,'curb':4}[label]
                    cell=[zone['zone_id'],ring,sector,round(max(0.2,min(span,6)),3),round(span,3),label_id,
                        round(float(sums[index]/counts[index]),3),int(counts[index])]
                else:
                    polygon=make_polar_annular_wedge(r0,r1,p0,p1)
                    cell=dict(polygon=polygon,elev=max(0.2,min(span,6)),delta_z=span,
                        z_mean=float(sums[index]/counts[index]),label=label,cx=cx,cy=cy,r=radius,
                        azimuth_deg=math.degrees(phi)%360,zone_id=zone['zone_id'],zone_name=zone['name'],
                        point_density=int(counts[index]))
                    if not np.array_equal(pose.translation,np.zeros(3)) or not np.array_equal(pose.rotation,np.eye(3)):
                        cell['world_polygon']=CoordinateTransformer.transform_cell_polygon(polygon,pose,to_world=True)
                cells.append(cell);zone_cells.append(cell)
            pts=int(counts[zone_slice].sum());cls=len(zone_cells)
            zones.append(dict(zone_id=zone['zone_id'],name=zone['name'],range_meters=f"{zone['r_min']}–{zone['r_max']} m",
                resolution=f"{zone['delta_r']} m radial × {360/zone['num_sectors']:.2f}° angular",
                accuracy='Not evaluated',points_ingested=pts,cells_generated=cls,
                compression_ratio_pct=100*(1-cls/max(pts,1)) if pts else 0,
                threats_detected=sum((c[5]==2 if compact else c['label']=='dynamic_target') for c in zone_cells),purpose=zone['purpose']))
        size=len(json.dumps(cells,separators=(',',':')).encode())
        radio={}
        for key,rate in [('cbr_64kbps',64000),('sdr_256kbps',256000),('link16_1200kbps',1200000)]:
            radio[key]=dict(link_name=f'{rate/1000:g} kbps link (simulation)',tx_time_3d_sec=total_points*16*8/rate,
                tx_time_foveated_ms=size*8/rate*1000,status='Estimated transfer only; no radio tested')
        report=dict(total_points=total_points,total_foveated_cells=len(cells),zones=zones,
            overall_compression_pct=100*(1-len(cells)/max(total_points,1)) if total_points else 0,tactical_radio_simulation=radio)
        return cells,report

class ThreatKinematicTracker:
    """Compatibility helper; no invented velocities or time-to-collision."""
    def update_threats(self,raw_threats,ego_speed_mps=0,dt=0.1):
        return [dict(threat,urgency_status='Motion not evaluated') for threat in raw_threats]
