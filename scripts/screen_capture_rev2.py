# screen capture program for Robot Learning
import numpy as np
import pygame 
from scipy.interpolate import UnivariateSpline
import h5py
from tkinter.filedialog import askopenfilename

def inbounds(x, y, rect):
    return (x > rect[0]) and (x < (rect[0] + rect[2])) and (y > rect[1]) and (y < (rect[1] + rect[3]))

class Screen_Capture(object):

    def __init__(self):
        #empty lists to store trajectory data
        self.demos = []
        self.smoothed_demos = []
        self.selected_demos = []
        self.norm_smoothed_demos = []
        
        #pygame constants
        self.width = 600
        self.height = 600
        
        self.demo_bg_color = pygame.Color(255, 255, 255)
        self.bg_color = pygame.Color(37, 22, 236)
        self.demo_color = pygame.Color(0, 0, 0)
        self.past_demo_color = pygame.Color(128, 128, 128)
        self.button_color = pygame.Color(170, 170, 170)
        
        self.freq = 100 #Hz recording frequency
        
        #amount to resample for smoothing
        self.num_resample = 1000
        
        #rects are a -> a + dx and b -> b + dy as [a, b, dx, dy]
        self.demo_window_coords = [100, 0, 500, 500]
        self.smooth_button_coords = [20, 20, 60, 40]
        self.raw_button_coords = [20, 80, 60, 40]
        self.save_button_coords = [20, 400, 60, 40]
        self.load_button_coords = [20, 460, 60, 40]
        self.quit_button_coords = [20, 520, 60, 40]
        self.selector_text_coords = [120, 520, 400, 40]
        
        
    def capture(self):
        pygame.init()
        
        self.sf = pygame.font.SysFont('arial',20) 
        self.smooth_text = self.sf.render('Smooth', True, self.demo_color)
        self.raw_text = self.sf.render('Raw', True, self.demo_color)
        self.save_text = self.sf.render('Save', True, self.demo_color)
        self.load_text = self.sf.render('Load', True, self.demo_color)
        self.quit_text = self.sf.render('Quit', True, self.demo_color)
        self.select_text = self.sf.render('Selected Demos', True, self.button_color)
        
        st_rect = self.smooth_text.get_rect(center=(self.smooth_button_coords[0] + (self.smooth_button_coords[2] // 2), self.smooth_button_coords[1] + (self.smooth_button_coords[3] // 2)))
        raw_rect = self.raw_text.get_rect(center=(self.raw_button_coords[0] + (self.raw_button_coords[2] // 2), self.raw_button_coords[1] + (self.raw_button_coords[3] // 2)))
        sv_rect = self.save_text.get_rect(center=(self.save_button_coords[0] + (self.save_button_coords[2] // 2), self.save_button_coords[1] + (self.save_button_coords[3] // 2)))
        ld_rect = self.load_text.get_rect(center=(self.load_button_coords[0] + (self.load_button_coords[2] // 2), self.load_button_coords[1] + (self.load_button_coords[3] // 2)))
        qt_rect = self.quit_text.get_rect(center=(self.quit_button_coords[0] + (self.quit_button_coords[2] // 2), self.quit_button_coords[1] + (self.quit_button_coords[3] // 2)))
        sel_rect = self.select_text.get_rect(center=(self.selector_text_coords[0] + (self.selector_text_coords[2] // 2), self.selector_text_coords[1] + (self.selector_text_coords[3] // 2)))
        
        screen = pygame.display.set_mode((self.width,self.height))
        pygame.display.set_caption('Demonstration Recorder')
        clock = pygame.time.Clock()
        
        #wipe screen
        screen.fill(self.bg_color)
        pygame.draw.rect(screen, self.button_color, self.smooth_button_coords)
        pygame.draw.rect(screen, self.button_color, self.raw_button_coords)
        pygame.draw.rect(screen, self.button_color, self.save_button_coords)
        pygame.draw.rect(screen, self.button_color, self.load_button_coords)
        pygame.draw.rect(screen, self.button_color, self.quit_button_coords)
        
        screen.blit(self.smooth_text, st_rect)
        screen.blit(self.raw_text, raw_rect)
        screen.blit(self.save_text, sv_rect)
        screen.blit(self.load_text, ld_rect)
        screen.blit(self.quit_text, qt_rect)
        screen.blit(self.select_text, (self.selector_text_coords[0], self.selector_text_coords[1] + 40))
        
        show_smooth = False
        color = self.demo_bg_color
        nd = 0
        
        text = ''
        
        while(True):
            
            pygame.draw.rect(screen, self.demo_bg_color, self.demo_window_coords)
            pygame.draw.rect(screen, color, self.selector_text_coords)
            
            #draw old demos grayed out
            if show_smooth:
                for i in self.selected_demos:
                    for j in range(len(self.smoothed_demos[i][0])):
                        c = int(150 // 1.1**i)
                        pd_color = pygame.Color(c, c, c)
                        pygame.draw.circle(screen, pd_color, (int(self.smoothed_demos[i][1][j]), int(self.smoothed_demos[i][2][j])), 3)
            else:
                for i in self.selected_demos:
                    for j in range(len(self.demos[i][0])):
                        c = int(150 // 1.1**i)
                        pd_color = pygame.Color(c, c, c)
                        pygame.draw.circle(screen, pd_color, (self.demos[i][1][j], self.demos[i][2][j]), 3)
            
            recording = False
            X = []
            Y = []
            T = []
            
            
            active = False
            
            while recording == False:
                pygame.draw.rect(screen, self.demo_bg_color, self.demo_window_coords)
                pygame.draw.rect(screen, color, self.selector_text_coords)
                self.selected_text = self.sf.render(str(self.selected_demos) + text, True, self.demo_color)
                screen.blit(self.selected_text, sel_rect)
                
                #draw old demos grayed out
                if show_smooth:
                    for i in self.selected_demos:
                        for j in range(len(self.smoothed_demos[i][0])):
                            c = int(150 // 1.1**i)
                            pd_color = pygame.Color(c, c, c)
                            pygame.draw.circle(screen, pd_color, (int(self.smoothed_demos[i][1][j]), int(self.smoothed_demos[i][2][j])), 3)
                else:
                    for i in self.selected_demos:
                        for j in range(len(self.demos[i][0])):
                            c = int(150 // 1.1**i)
                            pd_color = pygame.Color(c, c, c)
                            pygame.draw.circle(screen, pd_color, (self.demos[i][1][j], self.demos[i][2][j]), 3)
                            
                (x,y) = pygame.mouse.get_pos() #get mouse position
                
                if pygame.mouse.get_pressed() == (1, 0, 0):
                
                    if (inbounds(x, y, self.smooth_button_coords)):
                        show_smooth = True
                    if (inbounds(x, y, self.raw_button_coords)):
                        show_smooth = False
                        
                    if (inbounds(x, y, self.save_button_coords)):
                        self.save_demo_h5()
                        
                    if (inbounds(x, y, self.load_button_coords)):
                        filename = askopenfilename()
                        print(filename)
                        if (len(filename) > 2):
                            fp = h5py.File(filename, 'r')
                            smooth = fp.get('smoothed')
                            read_demos = np.array(list(smooth.keys())).astype(int).tolist()
                            fp.close()
                            self.demos = []
                            self.smoothed_demos = []
                            nd = 0
                            for demo_num in read_demos:
                                ret = read_demo_h5(filename, demo_num)
                                self.demos.append(ret[0])
                                self.smoothed_demos.append(ret[1])
                                self.selected_demos.append(nd)
                                nd += 1
                        
                    if (inbounds(x, y, self.quit_button_coords)):
                        pygame.display.quit()
                        pygame.quit() 
                        exit()
                            
                    if (inbounds(x, y, self.selector_text_coords)):
                        #this thing     # Toggle the active variable.
                        print('Please enter the demos you wish to have currently selected (example "0 2") and press enter')
                        active = True
                        color = self.button_color if active else self.demo_bg_color
                    else:
                        active = False
                            
                    if (inbounds(x, y, self.demo_window_coords)):
                        recording = True
                    
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.display.quit()
                        pygame.quit()  
                        exit()
                    if event.type == pygame.KEYDOWN:
                        if active:
                            if event.key == pygame.K_RETURN:
                                print(text)
                                self.selected_demos = np.array(text.split()).astype(int).tolist()
                                text = ''
                            elif event.key == pygame.K_BACKSPACE:
                                text = text[:-1]
                            else:
                                text += event.unicode
                
                #update screen and clock    
                pygame.display.update()
                clock.tick(self.freq)
                
            
            while recording:
                
                (x,y) = pygame.mouse.get_pos() #get mouse position
                
                if pygame.mouse.get_pressed() == (1, 0, 0):
                    #store position and time data
                    if (inbounds(x, y, self.demo_window_coords)):
                        T.append(pygame.time.get_ticks())
                        X.append(x)
                        Y.append(y)
                        #draw current position
                        pygame.draw.circle(screen, self.demo_color, (x, y), 3)
                    
                #check for end recording    
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        recording = False
                    if event.type == pygame.MOUSEBUTTONUP:
                        recording = False
                        traj = [T, X, Y]
                        self.demos.append(traj)
                        #smooth
                        tt = np.linspace(T[0], T[-1], self.num_resample)
                        spx = UnivariateSpline(T, X)
                        spy = UnivariateSpline(T, Y)
                        spx.set_smoothing_factor(999)
                        spy.set_smoothing_factor(999)
                        xx = spx(tt)
                        yy = spy(tt)
                        smooth_traj = [tt, xx, yy]
                        self.smoothed_demos.append(smooth_traj)
                        self.selected_demos.append(nd)
                        nd += 1
                        
                #update screen and clock    
                pygame.display.update()
                clock.tick(self.freq)
                
            
    def save_demo_h5(self):
        print('saving')
        #create a temporary h5 file in case user does not see prompt
        path = 'C:/Users/BH/Documents/GitHub/pearl_test_env/h5 files/'
        name = 'temp'
        fp = h5py.File(path + name + '.h5', 'w')
        for i in self.selected_demos:
            #get data
            t = self.demos[i][0]
            x = self.demos[i][1]
            y = self.demos[i][2]
            #smooth data
            tt = self.smoothed_demos[i][0]
            xx = self.smoothed_demos[i][1]
            yy = self.smoothed_demos[i][2]
            #normalized smooth data
            max_val = max(np.max(np.abs(xx)), np.max(np.abs(yy)))
            xxn = xx / max_val
            yyn = yy / max_val
            #save data
            fp.create_dataset('unsmoothed/' + str(i) + '/t', data = np.array(t))
            fp.create_dataset('unsmoothed/' + str(i) + '/x', data = np.array(x))
            fp.create_dataset('unsmoothed/' + str(i) + '/y', data = np.array(y))
            fp.create_dataset('smoothed/' + str(i) + '/t', data = np.array(tt))
            fp.create_dataset('smoothed/' + str(i) + '/x', data = np.array(xx))
            fp.create_dataset('smoothed/' + str(i) + '/y', data = np.array(yy))
            fp.create_dataset('normalized/' + str(i) + '/t', data = np.array(tt))
            fp.create_dataset('normalized/' + str(i) + '/x', data = np.array(xxn))
            fp.create_dataset('normalized/' + str(i) + '/y', data = np.array(yyn))
        fp.close() #file cleanup
        name = input('Please enter a name for the file (do not enter the trailing .h5): ')
        fp = h5py.File(path + name + '.h5', 'w')
        for i in self.selected_demos:
            #get data
            t = self.demos[i][0]
            x = self.demos[i][1]
            y = self.demos[i][2]
            #smooth data
            tt = self.smoothed_demos[i][0]
            xx = self.smoothed_demos[i][1]
            yy = self.smoothed_demos[i][2]
            #normalized smooth data
            max_val = max(np.max(np.abs(xx)), np.max(np.abs(yy)))
            xxn = xx / max_val
            yyn = yy / max_val
            #save data
            fp.create_dataset('unsmoothed/' + str(i) + '/t', data = np.array(t))
            fp.create_dataset('unsmoothed/' + str(i) + '/x', data = np.array(x))
            fp.create_dataset('unsmoothed/' + str(i) + '/y', data = np.array(y))
            fp.create_dataset('smoothed/' + str(i) + '/t', data = np.array(tt))
            fp.create_dataset('smoothed/' + str(i) + '/x', data = np.array(xx))
            fp.create_dataset('smoothed/' + str(i) + '/y', data = np.array(yy))
            fp.create_dataset('normalized/' + str(i) + '/t', data = np.array(tt))
            fp.create_dataset('normalized/' + str(i) + '/x', data = np.array(xxn))
            fp.create_dataset('normalized/' + str(i) + '/y', data = np.array(yyn))
        fp.close() #file cleanup
        
#to read a previously saved file and return results and/or plot
def read_demo_h5(fname, demo_num):
    #open file and navigate to contents
    fp = h5py.File(fname, 'r')
    print(list(fp.keys()))
    smooth = fp.get('smoothed')
    print(list(smooth.keys()))
    smooth_demo = smooth.get(str(demo_num))
    sm_t = np.array(smooth_demo.get('t'))
    sm_x = np.array(smooth_demo.get('x'))
    sm_y = np.array(smooth_demo.get('y'))
    unsmooth = fp.get('unsmoothed')
    unsmooth_demo = unsmooth.get(str(demo_num))
    unsm_t = np.array(unsmooth_demo.get('t'))
    unsm_x = np.array(unsmooth_demo.get('x'))
    unsm_y = np.array(unsmooth_demo.get('y'))
    norm = fp.get('normalized')
    norm_demo = norm.get(str(demo_num))
    norm_t = np.array(norm_demo.get('t'))
    norm_x = np.array(norm_demo.get('x'))
    norm_y = np.array(norm_demo.get('y'))
    fp.close()
    return [[sm_t, sm_x, sm_y], [unsm_t, unsm_x, unsm_y], [norm_t, norm_x, norm_y]]

#to read a previously saved file and return results and/or plot
def read_demo_h5_old(fname, demo_num):
    #open file and navigate to contents
    fp = h5py.File(fname, 'r')
    print(list(fp.keys()))
    smooth = fp.get('smoothed')
    print(list(smooth.keys()))
    smooth_demo = smooth.get(str(demo_num))
    sm_t = np.array(smooth_demo.get('t'))
    sm_x = np.array(smooth_demo.get('x'))
    sm_y = np.array(smooth_demo.get('y'))
    unsmooth = fp.get('unsmoothed')
    unsmooth_demo = unsmooth.get(str(demo_num))
    unsm_t = np.array(unsmooth_demo.get('t'))
    unsm_x = np.array(unsmooth_demo.get('x'))
    unsm_y = np.array(unsmooth_demo.get('y'))
    fp.close()
    return [[sm_t, sm_x, sm_y], [unsm_t, unsm_x, unsm_y]]

if __name__ == '__main__':
    obj = Screen_Capture()
    obj.capture()
    #read_demo_h5('test_sine.h5', 1)