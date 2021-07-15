names = {'PRESSING', 'PUSHING', 'REACHING', 'WRITING'};
num_users = 10;
%num_demos = 6;
for n=1:length(names)
    name = names{n};
    for u=1:num_users
        load([name '/' num2str(u) '.mat']);
        num_demos = length(dataset);
        for d=1:num_demos
            time = dataset(d).time;
            h5create(['RAIL_' name '.h5'], ['/user' num2str(u) '/demo' num2str(d) '/time'], size(time))
            h5write(['RAIL_' name '.h5'], ['/user' num2str(u) '/demo' num2str(d) '/time'], time)
            
            pos = dataset(d).pos;
            h5create(['RAIL_' name '.h5'], ['/user' num2str(u) '/demo' num2str(d) '/pos'], size(pos))
            h5write(['RAIL_' name '.h5'], ['/user' num2str(u) '/demo' num2str(d) '/pos'], pos)
            
            obj = dataset(d).obj;
            h5create(['RAIL_' name '.h5'], ['/user' num2str(u) '/demo' num2str(d) '/obj'], size(obj))
            h5write(['RAIL_' name '.h5'], ['/user' num2str(u) '/demo' num2str(d) '/obj'], obj)
            
            vel = dataset(d).vel;
            h5create(['RAIL_' name '.h5'], ['/user' num2str(u) '/demo' num2str(d) '/vel'], size(vel))
            h5write(['RAIL_' name '.h5'], ['/user' num2str(u) '/demo' num2str(d) '/vel'], vel)
            
            acc = dataset(d).acc;
            h5create(['RAIL_' name '.h5'], ['/user' num2str(u) '/demo' num2str(d) '/acc'], size(acc))
            h5write(['RAIL_' name '.h5'], ['/user' num2str(u) '/demo' num2str(d) '/acc'], acc)
        end
    end
end