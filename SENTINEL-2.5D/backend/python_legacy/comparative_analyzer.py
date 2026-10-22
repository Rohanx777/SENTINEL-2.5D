"""Measured representation sizes; network transfer times remain labelled estimates."""
import json
import numpy as np
class ComparativeMetrics:
    @staticmethod
    def compute(points_3d,cells_25d,**kwargs):
        raw=points_3d.nbytes;grid=len(json.dumps(cells_25d,separators=(',',':')).encode());rate=1_250_000
        def item(name,count,unit,size,fidelity):
            return dict(entity_type=name,count=count,unit=unit,raw_size_kb=size/1024,raw_size_mb=size/1048576,
                estimated_tx_time_ms=size/rate*1000,client_memory_mb=0,z_fidelity=fidelity,
                computation_load='Not measured',tactical_feasibility='10 Mbps ideal-link estimate; overhead excluded')
        return dict(summary=dict(compression_ratio_pct=100*(1-grid/max(raw,1)),bandwidth_reduction_factor=raw/max(grid,1),
                latency_improvement_ms=(raw-grid)/rate*1000),
            point_cloud_3d=item('Raw float32 scan',len(points_3d),'points',raw,'Full x,y,z,intensity'),
            grid_25d=item('2.5D JSON grid',len(cells_25d),'cells',grid,'Per-cell height range; vertical detail lost'),
            tradeoff_analysis=[dict(dimension='Byte size',winner='Measured per frame',description='Full diagnostic transport also includes preview and metadata; see total payload bytes.'),
                dict(dimension='Obstacle accuracy',winner='Not evaluated',description='Grid aggregation changes spatial detail; no equivalence or safety claim is made.')])

def sample_point_cloud_3d(points,labels=None,max_sample_points=1500):
    if not len(points): return []
    ids=np.linspace(0,len(points)-1,min(len(points),max_sample_points),dtype=int)
    names=['road','static_obstacle','dynamic_target']
    return [dict(x=float(points[i,0]),y=float(points[i,1]),z=float(points[i,2]),intensity=float(points[i,3]),
        label=names[int(labels[i])] if labels is not None else 'unknown',label_id=int(labels[i]) if labels is not None else 0) for i in ids]
