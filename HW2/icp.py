import random
from poseutil import *

if __name__=="__main__":
    voxel_size = 0.005
    correct_cnt = 0
    total_cnt = 0
    with open(os.path.join(split_dir, "val.txt"), 'r') as f:
        scenenames = [line.strip() for line in f if line.strip() and int(line.strip()[0]) <= 2]
    # sample 20% val data for evaluating
    # scenenames = random.sample(scenenames,int(0.2*len(scenenames)))

    for scenename in tqdm(scenenames):
        ids, pcds, scales, poses = scene_to_pcds(scenename)
        for i,id in enumerate(ids):
            source = get_open3d_PointCloud(canonical_pcds[id])
            target = get_open3d_PointCloud(pcds[id])
            #draw_registration_result(source,target,show_normal=False)

            source_down, source_fpfh = preprocess_point_cloud(source,voxel_size)
            target_down, target_fpfh = preprocess_point_cloud(target,voxel_size)
            fpfh_init = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
            #draw_registration_result(source_down, target_down, fpfh_init.transformation)

            source_down2, source_fpfh2 = preprocess_point_cloud(source,voxel_size)
            target_down2, target_fpfh2 = preprocess_point_cloud(target,voxel_size)
            fpfh_init2 = execute_global_registration(source_down2, target_down2, source_fpfh2, target_fpfh2, voxel_size)
            #draw_registration_result(source_down2, target_down2, fpfh_init2.transformation)

            reg_refine = open3d.pipelines.registration.registration_icp(
                source, target, voxel_size, fpfh_init.transformation, 
                open3d.pipelines.registration.TransformationEstimationPointToPlane(), 
                open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
            
            reg_refine2 = open3d.pipelines.registration.registration_icp(
                source, target, voxel_size, fpfh_init2.transformation, 
                open3d.pipelines.registration.TransformationEstimationPointToPlane(), 
                open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

            dt,dtheta=compare(poses[i],reg_refine.transformation,scales[id],id2symm[id])
            dt2,dtheta2=compare(poses[i],reg_refine2.transformation,scales[id],id2symm[id])

            if dtheta > dtheta2:
                #draw_registration_result(source,target,reg_refine2.transformation)
                dtheta = dtheta2
                dt = dt2
            else: pass#draw_registration_result(source,target,reg_refine.transformation)

            if abs(dt)<0.01 and dtheta<5:
                correct_cnt+=1
            total_cnt+=1

            
    print(f'Accuracy = {correct_cnt/total_cnt}')
    # final result over val: 0.6911408406194037