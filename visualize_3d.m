% To run the code: visualize_3d('runname')

function visualize_3d(runname)
    % VISUALIZE_3D  Read .dat files from PFC code and make 3D isosurface  plots
    %
    % Usage:
    %   visualize_3d('runname')
    %
    % This will read 'runname.in' for grid size and all 'runname-t:*.dat'  files,
    % then generate .jpg files with 3D isosurfaces of the smoothed field  u_n.

    %% --- Step 1: Read dimensions from input file (.in) ---
    infile = [runname, '.in'];
    fid = fopen(infile, 'r');
    if fid == -1
        error('Could not open input file %s', infile);
    end
    
    W = []; H = []; Z = [];
    while ~feof(fid)
        line = strtrim(fgetl(fid));
        if startsWith(line, 'A') % line like: A W H Z
            parts = strsplit(line);
            W = str2double(parts{2});
            H = str2double(parts{3});
            Z = str2double(parts{4});
            break;
        end
    end
    fclose(fid);

    if isempty(W)
        error('Could not find grid dimensions in %s', infile);
    end
    fprintf('Grid dimensions: W=%d, H=%d, Z=%d\n', W, H, Z);

    %% --- Step 2: Find .dat files ---
    files = dir([runname, '-t:*.dat']);
    if isempty(files)
        error('No .dat files found for runname %s', runname);
    end

    %% --- Step 3: Loop over .dat files and visualize ---
    for k = 1:length(files)
        fname = files(k).name;
        fprintf('Processing %s ...\n', fname);

        % Load .dat file
        data = load(fname);
        if numel(data) ~= W*H*Z*2
            error('Data size mismatch: expected %d rows', W*H*Z);
        end

        % Extract q and u_n, reshape into (Z,H,W)
        q = reshape(data(:,1), [W, H, Z]);
        u_n = reshape(data(:,2), [W, H, Z]);

        % MATLAB arrays are (row,col,page) = (y,x,z), so permute for consistency
        u_n = permute(u_n, [2 1 3]);

        % --- Step 4: Make isosurface plot ---
        isoValue = 0.0;   % change if needed
        figure('Visible','off');
        p = patch(isosurface(u_n, isoValue));
        isonormals(u_n, p);
        p.FaceColor = 'interp';
        p.EdgeColor = 'none';

        % Lighting & view
        camlight; lighting gouraud;
        axis equal off;
        colormap('inferno'); % requires MATLAB R2020b+, else use 'parula'
        title(sprintf('%s, isosurface = %.2f', fname, isoValue));

        % --- Step 5: Save as high-res JPG ---
        outname = strrep(fname, '.dat', '.jpg');
        print(outname, '-djpeg', '-r600'); % 600 DPI high resolution

        close;
        fprintf('Saved %s\n', outname);
    end
end

