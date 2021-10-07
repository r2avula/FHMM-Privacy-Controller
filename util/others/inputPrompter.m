function [input_string] = inputPrompter(wait_time)
if(nargin<1 || isempty(wait_time))
    wait_time = inf;
end
input_string = '';
if(isinf(wait_time))
    input_string = input(': ','s');
elseif(wait_time>0)
    t = timer;
    t.ExecutionMode = 'singleShot';
    t.StartDelay = wait_time;
    t.TimerFcn = @pressEnter;
    start(t)
    input_string = input(': ','s');
    stop(t);
    delete(t);
end
    function pressEnter(~,~)
        import java.awt.*;
        import java.awt.event.*;
        rob = Robot;
        rob.keyPress(KeyEvent.VK_N)
        rob.keyRelease(KeyEvent.VK_N)
        rob.keyPress(KeyEvent.VK_ENTER)
        rob.keyRelease(KeyEvent.VK_ENTER)
    end
end

