import cv2
import numpy as np
import multiprocessing

def Normalized_Img(img, Truncating = False):
    # This function works for both grayscale and RGB images
    # Arguments:
        #   img --- input image
        #   Truncating --- Truncating values more than 255 and less than 0 or not
    # Output:
        #   Image scaled between 0 and 255
        
    Image = img.copy()
    Image = np.float64(Image)
    
    if Truncating == True:
        Image[Image < 0] = 0
        Image[Image > 255] = 255
        
    try:
        for i in range(3):    
            Image[:,:,i] = Image[:,:,i] - np.min(Image[:,:,i])
            Image[:,:,i] = Image[:,:,i]/np.max(Image[:,:,i])*255
            
    except:
        Image = Image - np.min(Image)
        Image = Image/np.max(Image)*255  
    return Image    

def K_Smallest_Index(a, k):
    # Finding indices of the K smallest elements of an array
    reshaped_a = np.reshape(a, (1, a.shape[0]*a.shape[1]), order = 'F')[0]
    sorted_a =  np.sort(reshaped_a)[0:k]
    index = []
    for i in range(k):
        index.append(np.where(reshaped_a == sorted_a[i])[0][0])
    index = np.array(index)    
    return (index%a.shape[0], index//a.shape[0])


def Stitch_Vertical(template1, template2):
    # * find stitch location of two templates
    # * stitch template1 from right side to template2 from left side
    
    square_err  = pow(np.float64(template1[:,:,0]) - np.float64(template2[:,:,0]),2)
    square_err += pow(np.float64(template1[:,:,1]) - np.float64(template2[:,:,1]),2)
    square_err += pow(np.float64(template1[:,:,2]) - np.float64(template2[:,:,2]),2)
    
    E = np.zeros(square_err.shape)
    E[0,:] = square_err[0,:]
    for i in range(1,E.shape[0]):
        for j in range(E.shape[1]):
            if j == 0:
                E[i][j] = square_err[i][j] + min(E[i-1][j], E[i-1][j+1])
            elif j == E.shape[1]-1:
                E[i][j] = square_err[i][j] + min(E[i-1][j], E[i-1][j-1])
            else:
                E[i][j] = square_err[i][j] + min(E[i-1][j-1], E[i-1][j], E[i-1][j+1])
    
    stitch = np.zeros(E.shape[0], dtype = np.int64)
    stitch[-1] = np.int64(np.where(E[-1,:] == np.min(E[-1,:]))[0][0])
    j = stitch[-1]
    for i in range(E.shape[0]-2, -1, -1):
        if j == 0:
            if E[i][j] <= E[i][j+1]:
                stitch[i] = j
            else:
                j = j + 1
                stitch[i] = j
        
        elif j == E.shape[1] - 1:
            if E[i][j] <= E[i][j-1]:
                stitch[i] = j
            else:
                j = j - 1
                stitch[i] = j
        else:
            temp = min(E[i][j-1], E[i][j], E[i][j+1])
            if temp == E[i][j-1]:
                j = j - 1
                stitch[i] = j
            elif temp == E[i][j]:
                stitch[i] = j
            else:
                j = j + 1
                stitch[i] = j
    return stitch


def Stitch_Horizontal(template1, template2):
    # * find stitch location of two templates
    # * stitch template1 from down side to template2 from up side
   
    square_err  = pow(np.float64(template1[:,:,0]) - np.float64(template2[:,:,0]),2)
    square_err += pow(np.float64(template1[:,:,1]) - np.float64(template2[:,:,1]),2)
    square_err += pow(np.float64(template1[:,:,2]) - np.float64(template2[:,:,2]),2)
    
    
    E = np.zeros(square_err.shape)    
    E[:,0] = square_err[:,0]
    for i in range(E.shape[0]):
        for j in range(1,E.shape[1]):
            if i == 0:
                E[i][j] = square_err[i][j] + min(E[i][j-1], E[i+1][j-1])
            elif i == E.shape[0]-1:
                E[i][j] = square_err[i][j] + min(E[i][j-1], E[i-1][j-1])
            else:
                E[i][j] = square_err[i][j] + min(E[i-1][j-1], E[i][j-1], E[i+1][j-1])
                
        
    stitch = np.zeros(E.shape[1], dtype = np.int64)
    stitch[-1] = np.int64(np.where(E[:,-1] == np.min(E[:,-1]))[0][0])
    i = stitch[-1]
    for j in range(E.shape[1]-2, -1, -1):
        if i == 0:
            if E[i][j] <= E[i+1][j]:
                stitch[j] = i
            else:
                i = i + 1
                stitch[j] = i
            
        elif i == E.shape[0] - 1:
            if E[i][j] <= E[i-1][j]:
                stitch[j] = i
            else:
                i = i - 1
                stitch[j] = i
        else:
            temp =  min(E[i-1][j], E[i][j], E[i+1][j])
            if temp == E[i-1][j]:
                i = i - 1
                stitch[j] = i
            elif temp == E[i][j]:
                stitch[j] = i
            else:
                i = i + 1
                stitch[j] = i
    return stitch
    
def Collision_Detection(stitch1, stitch2, template_size):
    # . Detect Collision of a vertical and a horizontal min cut in a template
    # . We use "and" logic in order to detect the collision of these min cuts
    # . If the collision is not detected then we shift one of the stitches for one step and then we find collision!
    
    S1 = np.zeros((template_size,template_size))
    S2 = np.zeros((template_size,template_size))
    
    for i in range(S1.shape[0]):
        S1[i, stitch1[i]] = 1
        S2[stitch2[i], i] = 1    
    S = np.multiply(S1, S2)
    if np.max(S) != 0:    
        loc = np.where(S == np.max(S))
        max_dist = np.array(loc[0]**2 + loc[1]**2).argmax()
        Collision = (loc[0][max_dist], loc[1][max_dist])
        return Collision
    
    else: #if collision is not detected
        S1 = np.zeros((template_size,template_size))
        for i in range(1, S1.shape[0]):   # Shift the stitch
            S1[i, stitch1[i-1]] = 1       
        S = np.multiply(S1, S2)
        loc = np.where(S == np.max(S))
        max_dist = np.array(loc[0]**2 + loc[1]**2).argmax()
        Collision = (loc[0][max_dist], loc[1][max_dist])
        return Collision   


def Texture_Transfer(source, target, patch_size=30, overlap_size=10,
                     alpha=0.1, randomness=100):
    try:
        transfer_img = np.zeros(target.shape)
        Pointer_right = patch_size
        Pointer_down  = patch_size
        
        rand_loc1 = np.random.randint(source.shape[0]-patch_size)
        rand_loc2 = np.random.randint(source.shape[1]-patch_size)
        
        transfer_img[:patch_size, :patch_size, :] = source[rand_loc1:rand_loc1+patch_size,
                                                           rand_loc2:rand_loc2+patch_size, :]
        
        while Pointer_right < transfer_img.shape[1]:
        
            
            template1 = transfer_img[:Pointer_down,
                                    Pointer_right-overlap_size:Pointer_right,
                                    :]
            template2 = target[:Pointer_down,
                               Pointer_right-overlap_size:Pointer_right,
                               :]
            
            map1 = cv2.matchTemplate(source[:-patch_size,:-patch_size,:], np.uint8(template1), cv2.TM_SQDIFF)
            map2 = cv2.matchTemplate(source[:-patch_size,:-patch_size,:], np.uint8(template2), cv2.TM_SQDIFF)
            
            map3 = map1 * (alpha) + map2 * (1-alpha)
            
        
            min_locs = K_Smallest_Index(map3, randomness) #choose randomly between candidaties!
            temp = np.random.randint(1, randomness)
            loc = (min_locs[0][temp], min_locs[1][temp])
            
            # patch extracted from prior image
            nxt_patch = source[loc[0]:loc[0]+patch_size,
                               loc[1]:loc[1]+patch_size,:]
            
            # the template from nxtpatch used for min cut
            nxt_template = nxt_patch[:,0:overlap_size,:]
            
            # stitch location ...
            stitch = Stitch_Vertical(template1, nxt_template)
            
            # the final patch with considering min cuts and so on
            overlap = np.zeros(template1.shape, dtype = np.uint8)
            for i in range(stitch.shape[0]):
                overlap[i,0:stitch[i],:] = template1[i,0:stitch[i],:]
                overlap[i,stitch[i]:,:]  = nxt_template[i,stitch[i]:,:]
            
            # place "overlap" in synthesized image
            transfer_img[0:patch_size,
                            Pointer_right-overlap_size:Pointer_right,
                            :] = overlap
            try:
                transfer_img[0:patch_size,
                                Pointer_right:Pointer_right+patch_size-overlap_size,
                                :] = nxt_patch[:, overlap_size:, :]
            except:
                transfer_img[0:patch_size,
                                Pointer_right:,
                                :] = nxt_patch[:,
                                               overlap_size:overlap_size+transfer_img.shape[1]-Pointer_right,
                                               :]
            Pointer_right = Pointer_right + patch_size - overlap_size
        
        Pointer_right = patch_size
        template1 = transfer_img[Pointer_down-overlap_size:Pointer_down,:Pointer_right,:]
        template2 = target[Pointer_down-overlap_size:Pointer_down,:Pointer_right,:]
        map1 = cv2.matchTemplate(source[:-patch_size,:-patch_size,:], np.uint8(template1), cv2.TM_SQDIFF)
        map2 = cv2.matchTemplate(source[:-patch_size,:-patch_size,:], np.uint8(template2), cv2.TM_SQDIFF)
        map3 = map1 * (alpha) + map2 * (1-alpha)
        
        min_locs = K_Smallest_Index(map3, randomness)
        temp = np.random.randint(1, randomness)
        loc = (min_locs[0][temp], min_locs[1][temp])
        
        # patch extracted from prior image
        nxt_patch = source[loc[0]:loc[0]+patch_size,
                           loc[1]:loc[1]+patch_size,:]
        nxt_template = nxt_patch[0:overlap_size,:,:] 
        
        
        stitch = Stitch_Horizontal(template1, nxt_template)
        
        overlap = np.zeros(template1.shape, dtype = np.uint8)
        for i in range(stitch.shape[0]):
            overlap[0:stitch[i],i,:] = template1[0:stitch[i],i,:]
            overlap[stitch[i]:,i,:]  = nxt_template[stitch[i]:,i,:]
        
        transfer_img[Pointer_down-template1.shape[0]:Pointer_down,
                     Pointer_right-template1.shape[1]:Pointer_right,
                     :] = overlap
        
        transfer_img[Pointer_down:Pointer_down+patch_size-template1.shape[0],
                     Pointer_right-template1.shape[1]:Pointer_right
                     :] = nxt_patch[template1.shape[0]:,:, :]
        

        while Pointer_down < transfer_img.shape[0]:
            # plt.imshow(np.uint8(Normalized_Img(transfer_img)))
            # plt.show()
            while Pointer_right <= transfer_img.shape[1] - (patch_size - overlap_size):
                template1 = np.zeros((patch_size, patch_size, 3), dtype = np.uint8)
                template1[0:overlap_size,:,:] = transfer_img[Pointer_down-overlap_size:Pointer_down,
                                                             Pointer_right-overlap_size:Pointer_right-overlap_size+patch_size,
                                                             :] 
                template1[:,0:overlap_size,:] = transfer_img[Pointer_down-overlap_size:Pointer_down-overlap_size+patch_size,
                                                             Pointer_right-overlap_size:Pointer_right,
                                                             :]
            
                template2 = np.zeros((patch_size, patch_size, 3), dtype = np.uint8)
                template2[0:overlap_size,:,:] = target[Pointer_down-overlap_size:Pointer_down,
                                                       Pointer_right-overlap_size:Pointer_right-overlap_size+patch_size,
                                                       :] 
                template2[:,0:overlap_size,:] = target[Pointer_down-overlap_size:Pointer_down-overlap_size+patch_size,
                                                       Pointer_right-overlap_size:Pointer_right,
                                                       :]
            
            
            
                # Mask used for template matching
                mask_template = np.ones((patch_size, patch_size, 3), dtype=np.uint8)
                mask_template[overlap_size:, overlap_size:,:] = np.zeros((patch_size-overlap_size,patch_size-overlap_size, 3), dtype=np.uint8)
            
            
                map1 = cv2.matchTemplate(source, template1, cv2.TM_SQDIFF, mask = mask_template)
                map2 = cv2.matchTemplate(source, template2, cv2.TM_SQDIFF, mask = mask_template)
                map3 = map1 * (alpha) + map2 * (1-alpha)
                min_locs = K_Smallest_Index(map3, randomness)
                temp = np.random.randint(1, randomness)
                loc = (min_locs[0][temp], min_locs[1][temp])
                nxt_patch = source[loc[0]:loc[0]+patch_size,
                                   loc[1]:loc[1]+patch_size,:]
                
                
                stitch1 = Stitch_Vertical(template1[:,0:overlap_size],
                                          nxt_patch[:,0:overlap_size]) 
                
                stitch2 = Stitch_Horizontal(template1[0:overlap_size,:],
                                            nxt_patch[0:overlap_size,:]) 
                
                collision = Collision_Detection(stitch1, stitch2, overlap_size)
                
                overlap = nxt_patch.copy()
                for i in range(collision[0], patch_size):
                    overlap[i,0:stitch1[i],:] = template1[i,0:stitch1[i],:]
                    
                for i in range(collision[1], patch_size):
                    overlap[0:stitch2[i],i,:] = template1[0:stitch2[i],i,:]
                
                overlap[0:collision[0],0:collision[1],:] = template1[0:collision[0],0:collision[1],:]    
                overlap[overlap_size:,overlap_size:,:] = nxt_patch[overlap_size:,overlap_size:,:]
                
                try:
                    transfer_img[Pointer_down-overlap_size:Pointer_down-overlap_size+patch_size,
                                 Pointer_right-overlap_size:Pointer_right-overlap_size+patch_size,
                                 :] = overlap
                except:
                    transfer_img[Pointer_down-overlap_size:Pointer_down-overlap_size+patch_size,
                                 Pointer_right-overlap_size:,
                                 :] = overlap[:,
                                              0:transfer_img.shape[1]-(Pointer_right-overlap_size),
                                              :]
                        
                
                Pointer_right = Pointer_right - overlap_size + patch_size
            
            
            
            
            template1 = np.zeros((patch_size, transfer_img.shape[1]-Pointer_right+overlap_size, 3), dtype = np.uint8)
            template1[0:overlap_size,:,:] = transfer_img[Pointer_down-overlap_size:Pointer_down,
                                                         Pointer_right-overlap_size:,
                                                         :] 
            template1[:,0:overlap_size,:] = transfer_img[Pointer_down-overlap_size:Pointer_down-overlap_size+patch_size,
                                                         Pointer_right-overlap_size:Pointer_right,
                                                         :]
            
            template2 = np.zeros((patch_size, transfer_img.shape[1]-Pointer_right+overlap_size, 3), dtype = np.uint8)
            template2[0:overlap_size,:,:] = target[Pointer_down-overlap_size:Pointer_down,
                                                   Pointer_right-overlap_size:,
                                                   :] 
            template2[:,0:overlap_size,:] = target[Pointer_down-overlap_size:Pointer_down-overlap_size+patch_size,
                                                   Pointer_right-overlap_size:Pointer_right,
                                                   :]
            
            mask_template = np.ones(template1.shape, dtype=np.uint8)
            mask_template[overlap_size:, overlap_size:,:] = 0#np.zeros((patch_size-overlap_size,patch_size-overlap_size, 3), dtype=np.uint8)
            
            
            map1 = cv2.matchTemplate(source, template1, cv2.TM_SQDIFF, mask = mask_template)
            map2 = cv2.matchTemplate(source, template2, cv2.TM_SQDIFF, mask = mask_template)
            map3 = map1 * (alpha) + map2 * (1-alpha)
            min_locs = K_Smallest_Index(map3, randomness)
            temp = np.random.randint(1, randomness)
            loc = (min_locs[0][temp], min_locs[1][temp])
            nxt_patch = source[loc[0]:loc[0]+template1.shape[0],
                               loc[1]:loc[1]+template1.shape[1],:]
            
            
            stitch1 = Stitch_Vertical(template1[:,0:overlap_size],
                                      nxt_patch[:,0:overlap_size]) 
            
            stitch2 = Stitch_Horizontal(template1[0:overlap_size,:],
                                        nxt_patch[0:overlap_size,:]) 
            
            collision = Collision_Detection(stitch1, stitch2, overlap_size)
            
            overlap = nxt_patch.copy()
            for i in range(collision[0], template1.shape[0]):
                overlap[i,0:stitch1[i],:] = template1[i,0:stitch1[i],:]
                
            for i in range(collision[1], template1.shape[1]):
                overlap[0:stitch2[i],i,:] = template1[0:stitch2[i],i,:]
            
            overlap[0:collision[0],0:collision[1],:] = template1[0:collision[0],0:collision[1],:]    
            overlap[overlap_size:,overlap_size:,:] = nxt_patch[overlap_size:,overlap_size:,:]
            
            transfer_img[Pointer_down-overlap_size:Pointer_down-overlap_size+patch_size,
                         Pointer_right-overlap_size:,
                         :] = overlap

            
            # New Line texture synthesis Again!
            Pointer_right = patch_size
            Pointer_down = Pointer_down - overlap_size + patch_size
        
            
            template1 = transfer_img[Pointer_down-overlap_size:Pointer_down,
                                    :Pointer_right, :]
            template2 = target[Pointer_down-overlap_size:Pointer_down,
                               :Pointer_right, :]
            
            map1 = cv2.matchTemplate(source[:-patch_size,:-patch_size,:],
                                     np.uint8(template1), cv2.TM_SQDIFF)
            map2 = cv2.matchTemplate(source[:-patch_size,:-patch_size,:],
                                     np.uint8(template2), cv2.TM_SQDIFF)
            map3 = map1 * (alpha) + map2 * (1-alpha)
            
            min_locs = K_Smallest_Index(map3, randomness)
            temp = np.random.randint(1, randomness)
            loc = (min_locs[0][temp], min_locs[1][temp])
            nxt_patch = source[loc[0]:loc[0]+patch_size,
                               loc[1]:loc[1]+patch_size,:]
        
            nxt_template = nxt_patch[0:template1.shape[0],:,:]    
            
            
            stitch = Stitch_Horizontal(template1, nxt_template)
            
            
            overlap = np.zeros(template1.shape, dtype = np.uint8)
            for i in range(stitch.shape[0]):
                overlap[0:stitch[i],i,:] = template1[0:stitch[i],i,:]
                overlap[stitch[i]:,i,:]  = nxt_template[stitch[i]:,i,:]
            
            transfer_img[Pointer_down-template1.shape[0]:Pointer_down,
                         Pointer_right-template1.shape[1]:Pointer_right,
                         :] = overlap
            if Pointer_down < transfer_img.shape[0]:  
                try:
                    transfer_img[Pointer_down:Pointer_down+patch_size-template1.shape[0],
                                 Pointer_right-template1.shape[1]:Pointer_right
                                 :] = nxt_patch[template1.shape[0]:,:, :]
                except:
                    transfer_img[Pointer_down:,
                                 Pointer_right-template1.shape[1]:Pointer_right
                                 :] = nxt_patch[0:transfer_img.shape[0]-Pointer_down, :, :]
        return transfer_img            
    
    except:
        try:
            # while Pointer_right < transfer_img.shape[1]:
            while Pointer_right <= transfer_img.shape[1] - (patch_size - overlap_size):   
                template1 = np.zeros((transfer_img.shape[0]-Pointer_down+overlap_size, patch_size, 3), dtype = np.uint8)
                template1[0:overlap_size,:,:] = transfer_img[Pointer_down-overlap_size:Pointer_down,
                                                             Pointer_right-overlap_size:Pointer_right-overlap_size+patch_size,
                                                             :] 
                template1[:,0:overlap_size,:] = transfer_img[Pointer_down-overlap_size:,
                                                             Pointer_right-overlap_size:Pointer_right,
                                                             :]
                
                template2 = np.zeros((transfer_img.shape[0]-Pointer_down+overlap_size, patch_size, 3), dtype = np.uint8)
                template2[0:overlap_size,:,:] = target[Pointer_down-overlap_size:Pointer_down,
                                                       Pointer_right-overlap_size:Pointer_right-overlap_size+patch_size,
                                                       :] 
                template2[:,0:overlap_size,:] = target[Pointer_down-overlap_size:,
                                                       Pointer_right-overlap_size:Pointer_right,
                                                       :]
                
                mask_template = np.ones(template1.shape, dtype=np.uint8)
                mask_template[overlap_size:, overlap_size:,:] = 0#np.zeros((patch_size-overlap_size,patch_size-overlap_size, 3), dtype=np.uint8)
                
                map1 = cv2.matchTemplate(source, template1, cv2.TM_SQDIFF, mask = mask_template)
                map2 = cv2.matchTemplate(source, template2, cv2.TM_SQDIFF, mask = mask_template)
                map3 = map1 * (alpha) + map2 * (1-alpha)
                min_locs = K_Smallest_Index(map3, randomness)
                temp = np.random.randint(1, randomness)
                loc = (min_locs[0][temp], min_locs[1][temp])
                nxt_patch = source[loc[0]:loc[0]+template1.shape[0],
                                   loc[1]:loc[1]+template1.shape[1],:]
                
                
                stitch1 = Stitch_Vertical(template1[:,0:overlap_size],
                                          nxt_patch[:,0:overlap_size]) 
                
                stitch2 = Stitch_Horizontal(template1[0:overlap_size,:],
                                            nxt_patch[0:overlap_size,:]) 
                
                collision = Collision_Detection(stitch1, stitch2, overlap_size)
                
                
                overlap = nxt_patch.copy()
                for i in range(collision[0], template1.shape[0]):
                    overlap[i,0:stitch1[i],:] = template1[i,0:stitch1[i],:]
                    
                for i in range(collision[1], template1.shape[1]):
                    overlap[0:stitch2[i],i,:] = template1[0:stitch2[i],i,:]
                
                overlap[0:collision[0],0:collision[1],:] = template1[0:collision[0],0:collision[1],:]    
                overlap[overlap_size:,overlap_size:,:] = nxt_patch[overlap_size:,overlap_size:,:]
                transfer_img[Pointer_down-overlap_size:,
                             Pointer_right-overlap_size:Pointer_right-overlap_size+patch_size,
                             :] = overlap
                
                Pointer_right = Pointer_right - overlap_size + patch_size
                
                
                
            template1 = np.zeros((transfer_img.shape[0]-Pointer_down+overlap_size,
                                  transfer_img.shape[1]-Pointer_right+overlap_size,
                                  3), dtype = np.uint8)
            
            template1[0:overlap_size,:,:] = transfer_img[Pointer_down-overlap_size:Pointer_down,
                                                         Pointer_right-overlap_size:,
                                                         :] 
            template1[:,0:overlap_size,:] = transfer_img[Pointer_down-overlap_size:,
                                                         Pointer_right-overlap_size:Pointer_right,
                                                         :]
                
            template2 = np.zeros((transfer_img.shape[0]-Pointer_down+overlap_size,
                                  transfer_img.shape[1]-Pointer_right+overlap_size,
                                  3), dtype = np.uint8)
            
            template2[0:overlap_size,:,:] = target[Pointer_down-overlap_size:Pointer_down,
                                                   Pointer_right-overlap_size:,
                                                   :] 
            template2[:,0:overlap_size,:] = target[Pointer_down-overlap_size:,
                                                   Pointer_right-overlap_size:Pointer_right,
                                                   :]
                
            mask_template = np.ones(template1.shape, dtype=np.uint8)
            mask_template[overlap_size:, overlap_size:,:] = 0
            map1 = cv2.matchTemplate(source, template1, cv2.TM_SQDIFF, mask = mask_template)
            map2 = cv2.matchTemplate(source, template2, cv2.TM_SQDIFF, mask = mask_template)
            map3 = map1 * (alpha) + map2 * (1-alpha)
            min_locs = K_Smallest_Index(map3, randomness)
            temp = np.random.randint(1, randomness)
            loc = (min_locs[0][temp], min_locs[1][temp])
            nxt_patch = source[loc[0]:loc[0]+template1.shape[0],
                               loc[1]:loc[1]+template1.shape[1],:]
            
                
            stitch1 = Stitch_Vertical(template1[:,0:overlap_size],
                                      nxt_patch[:,0:overlap_size]) 
                
            stitch2 = Stitch_Horizontal(template1[0:overlap_size,:],
                                        nxt_patch[0:overlap_size,:]) 
                
            collision = Collision_Detection(stitch1, stitch2, overlap_size)
                
                
            overlap = nxt_patch.copy()
            for i in range(collision[0], template1.shape[0]):
                overlap[i,0:stitch1[i],:] = template1[i,0:stitch1[i],:]
            
            for i in range(collision[1], template1.shape[1]):
                overlap[0:stitch2[i],i,:] = template1[0:stitch2[i],i,:]
            
            overlap[0:collision[0],0:collision[1],:] = template1[0:collision[0],0:collision[1],:]    
            overlap[overlap_size:,overlap_size:,:] = nxt_patch[overlap_size:,overlap_size:,:]
            transfer_img[Pointer_down-overlap_size:,
                         Pointer_right-overlap_size:,
                         :] = overlap                
                
                
                
            return transfer_img
        
        except:
            return transfer_img 
    

if __name__ == '__main__':
    source = cv2.imread("source.jpg")
    target = cv2.imread('target.jpg')
    
    transfer_img = np.zeros(target.shape)
    
    patch_size = 50
    overlap_size = 30
    alpha = 0.1
    randomness = 100
    
    for iteration in range(3):
        SOURCE = [source, source, source, source]
        TARGET = [target, target, target, target]
        RESULT = []
        
        pool = multiprocessing.Pool(4)
        funclist = []
        for i in range(len(SOURCE)):
            f = pool.apply_async(Texture_Transfer, [SOURCE[i], TARGET[i], patch_size, overlap_size, alpha, randomness])
            funclist.append(f)
            
        for i, f in enumerate(funclist):
            RESULT.append(f.get(timeout = 1000))
        
        for i in range(len(RESULT)):
            transfer_img += RESULT[i]
            
    cv2.imwrite('Result.jpg', np.uint8(Normalized_Img(transfer_img)))
                
    