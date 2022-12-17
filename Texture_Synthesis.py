import cv2
import numpy as np
import matplotlib.pyplot as plt

def K_Smallest_Index(a, k):
    # Finding indexes of the K smallest elements of an array
    
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
    
    square_err = pow(np.float64(template1[:,:,0]) - np.float64(template2[:,:,0]),2)
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
   
    square_err = pow(np.float64(template1[:,:,0]) - np.float64(template2[:,:,0]),2)
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



def Texture_Synthesis(img, Texture_size, patch_size, overlap_size, randomness):
    # Creating syntehsized texture
    # input:
    # --- img: texture
    # --- Texture_size: size of the final image
    # --- patch_size: size of the patch we are about to merge
    # --- overlap_size: size of overlap between each two consecutive patches for stitching
    # --- randomness: an integer indicating the randomness of choosing the best patches
    
    Final_img_size = Texture_size
    
    patch_rows = patch_size
    patch_cols = patch_size
    template_cols = overlap_size
    
    # calculate size of image respect to patch size and the user's defined size
    Synthesized_img_size = np.ceil((Final_img_size - patch_cols)/(patch_cols - template_cols))
    Synthesized_img_size = Synthesized_img_size*(patch_cols - template_cols) + patch_cols
    Synthesized_img_size = np.int64(Synthesized_img_size)
    
    Synthesized_img = np.zeros((Synthesized_img_size,Synthesized_img_size,3), dtype = np.uint8)
    
    # first patch location chosen randomly
    random_loc = np.random.randint(1, [img.shape[0] - patch_rows,
                                       img.shape[1] - patch_cols])
    
    patch = img[random_loc[0]:random_loc[0]+patch_rows,
                random_loc[1]:random_loc[1]+patch_cols,:]
    Synthesized_img[0:patch_rows, 0:patch_cols, :] = patch
    
    # Pointers used in order to don't lose our location in sythesized img
    Pointer_right = 0
    Pointer_down = 0
    Pointer_right = Pointer_right + patch_cols
    Pointer_down = Pointer_down + patch_rows
    
    # while we haven't reached to the end of one line
    while Pointer_right <= Synthesized_img.shape[1]:
        
        # template used for template matching
        template = patch[:,-template_cols-1:-1,:]
        
        res = cv2.matchTemplate(img[0:-patch_rows,0:-patch_rows,:], template, cv2.TM_SQDIFF)
        min_locs = K_Smallest_Index(res, randomness) #choose randomly between candidaties!
        temp = np.random.randint(1, randomness)
        loc = (min_locs[0][temp], min_locs[1][temp])
        
        # patch extracted from prior image
        nxt_patch = img[loc[0]:loc[0]+patch_rows,
                        loc[1]:loc[1]+patch_cols,:]
        
        # the template from nxtpatch used for min cut
        nxt_template = nxt_patch[:,0:template.shape[1],:]
        
        # stitch location ...
        stitch = Stitch_Vertical(template, nxt_template)
        
        # the final patch with considering min cuts and so on
        overlap = np.zeros(template.shape, dtype = np.uint8)
        for i in range(stitch.shape[0]):
            overlap[i,0:stitch[i],:] = template[i,0:stitch[i],:]
            overlap[i,stitch[i]:,:] = nxt_template[i,stitch[i]:,:]
        
        # plcae "overlap" in synthesized image
        Synthesized_img[0:patch_rows,
                        Pointer_right-template.shape[1]:Pointer_right,
                        :] = overlap
        try:
            Synthesized_img[0:patch_rows,
                            Pointer_right:Pointer_right+patch_cols-template.shape[1],
                            :] = nxt_patch[:, template.shape[1]:, :]
        except:
            Synthesized_img[0:patch_rows,
                            Pointer_right:,
                            :] = nxt_patch[:,
                                           template.shape[1]:template.shape[1]+Synthesized_img.shape[1]-Pointer_right,
                                           :]
    
        Pointer_right = Pointer_right + patch_cols - template.shape[1]
        patch = nxt_patch
    
    # next patch have to be merged in the first of the new line
    Pointer_right = patch_cols # Reseting Pointers
    
    
    patch = Synthesized_img[Pointer_down-patch_rows:Pointer_down,
                            Pointer_right-patch_cols:Pointer_right,:]
    
    # template used for template matching  
    template = patch[-template_cols-1:-1,:,:]            
    res = cv2.matchTemplate(img[0:patch_rows,0:patch_rows,:], template, cv2.TM_SQDIFF)
    min_locs = K_Smallest_Index(res, randomness)
    temp = np.random.randint(1, randomness)
    loc = (min_locs[0][temp], min_locs[1][temp])
    
    # patch extracted from prior image
    nxt_patch = img[loc[0]:loc[0]+patch_rows,
                    loc[1]:loc[1]+patch_cols,:]
    nxt_template = nxt_patch[0:template.shape[0],:,:] 
    
    
    stitch = Stitch_Horizontal(template, nxt_template)
    
    overlap = np.zeros(template.shape, dtype = np.uint8)
    for i in range(stitch.shape[0]):
        overlap[0:stitch[i],i,:] = template[0:stitch[i],i,:]
        overlap[stitch[i]:,i,:] = nxt_template[stitch[i]:,i,:]
    
    Synthesized_img[Pointer_down-template.shape[0]:Pointer_down,
                    Pointer_right-template.shape[1]:Pointer_right,
                    :] = overlap
    
    Synthesized_img[Pointer_down:Pointer_down+patch_rows-template.shape[0],
                    Pointer_right-template.shape[1]:Pointer_right
                   :] = nxt_patch[template.shape[0]:,:, :]
    
    # Now we have to find L in image
    while Pointer_down < Synthesized_img.shape[0]:
        while Pointer_right <= Synthesized_img.shape[1] - (patch_cols - template_cols):
            template = np.zeros((patch_rows, patch_cols, 3), dtype = np.uint8)
            template[0:template_cols,:,:] = Synthesized_img[Pointer_down-template_cols:Pointer_down,
                                                            Pointer_right-template_cols:Pointer_right-template_cols+patch_cols,
                                                            :] 
            template[:,0:template_cols,:] = Synthesized_img[Pointer_down-template_cols:Pointer_down-template_cols+patch_rows,
                                                            Pointer_right-template_cols:Pointer_right,
                                                            :]
    
            # Mask used for template matching
            mask_template = np.ones(patch.shape, dtype=np.uint8)
            mask_template[template_cols:, template_cols:,:] = np.zeros((patch_cols-template_cols,patch_rows - template_cols, 3),dtype=np.uint8)
            res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF, mask = mask_template)
            min_locs = K_Smallest_Index(res, randomness)
            temp = np.random.randint(1, randomness)
            loc = (min_locs[0][temp], min_locs[1][temp])
            nxt_patch = img[loc[0]:loc[0]+patch_rows,
                            loc[1]:loc[1]+patch_cols,:]
            
            
            stitch1 = Stitch_Vertical(template[:,0:template_cols],
                                      nxt_patch[:,0:template_cols]) 
            
            stitch2 = Stitch_Horizontal(template[0:template_cols,:],
                                        nxt_patch[0:template_cols,:]) 
            
            collision = Collision_Detection(stitch1, stitch2, template_cols)
            
            overlap = nxt_patch.copy()
            for i in range(collision[0], patch_rows):
                overlap[i,0:stitch1[i],:] = template[i,0:stitch1[i],:]
                
            for i in range(collision[1], patch_cols):
                overlap[0:stitch2[i],i,:] = template[0:stitch2[i],i,:]
            
            overlap[0:collision[0],0:collision[1],:] = template[0:collision[0],0:collision[1],:]    
            overlap[template_cols:,template_cols:,:] = nxt_patch[template_cols:,template_cols:,:]
            
            try:
                Synthesized_img[Pointer_down-template_cols:Pointer_down-template_cols+patch_rows,
                                Pointer_right-template_cols:Pointer_right-template_cols+patch_cols,
                                :] = overlap
            except:
                Synthesized_img[Pointer_down-template_cols:Pointer_down-template_cols+patch_rows,
                                Pointer_right-template_cols:,
                                :] = overlap[:,
                                             0:Synthesized_img.shape[1]-(Pointer_right-template_cols),
                                             :]
                    
            
            Pointer_right = Pointer_right - template_cols + patch_cols
        
        # New Line texture synthesis Again!
        Pointer_right = patch_cols
        Pointer_down = Pointer_down - template_cols + patch_cols
        patch = Synthesized_img[Pointer_down-patch_rows:Pointer_down,
                                Pointer_right-patch_cols:Pointer_right,:]
        template = patch[-template_cols-1:-1,:,:]    
        res = cv2.matchTemplate(img[0:-patch_rows,0:-patch_rows,:], template, cv2.TM_SQDIFF)
        min_locs = K_Smallest_Index(res, randomness)
        temp = np.random.randint(1, randomness)
        loc = (min_locs[0][temp], min_locs[1][temp])
        nxt_patch = img[loc[0]:loc[0]+patch_rows,
                        loc[1]:loc[1]+patch_cols,:]
    
        nxt_template = nxt_patch[0:template.shape[0],:,:]    
        
        
        stitch = Stitch_Horizontal(template, nxt_template)
        
        
        overlap = np.zeros(template.shape, dtype = np.uint8)
        for i in range(stitch.shape[0]):
            overlap[0:stitch[i],i,:] = template[0:stitch[i],i,:]
            overlap[stitch[i]:,i,:] = nxt_template[stitch[i]:,i,:]
        
        Synthesized_img[Pointer_down-template.shape[0]:Pointer_down,
                        Pointer_right-template.shape[1]:Pointer_right,
                        :] = overlap
        if Pointer_down < Synthesized_img.shape[0]:  
            Synthesized_img[Pointer_down:Pointer_down+patch_rows-template.shape[0],
                            Pointer_right-template.shape[1]:Pointer_right
                           :] = nxt_patch[template.shape[0]:,:, :]
    
    # Blurrimg Image to have a better texture synthesis!
    Final_img = cv2.GaussianBlur(Synthesized_img[0:Final_img_size,
                                                 0:Final_img_size,
                                                 :], (3,3), 0.5)
    return Final_img


def Make_Result(img, Synthesized_img):
    # Merging the image and its synthesized form and plot them in another image
    
    Final_img = np.zeros((Synthesized_img.shape[0],
                          img.shape[1] + Synthesized_img.shape[1]+20, 3))
    
    Final_img[0:img.shape[0], 0:img.shape[1],:] = img.copy()
    Final_img[0:Synthesized_img.shape[0], img.shape[1]+20:, :] = Synthesized_img.copy()    
    return Final_img



if __name__ == '__main__':
    #loading your texture
    img = cv2.imread('texture.jpg') 
    Synthesized_img = Texture_Synthesis(img,
                                        Texture_size = 2500,
                                        patch_size = 200,
                                        overlap_size = 50,
                                        randomness = 50)
    Final_img = Make_Result(img, Synthesized_img)
    cv2.imwrite('result.jpg', Final_img)