stress = [28 -48 0;
          -48 0 0;
          0 0 0];
p_stress = eig(stress);


max_normal = max(abs(p_stress))
max_shear = 0.5 * abs(max(p_stress) - min(p_stress))
max_von_mises = sqrt(0.5 * ((p_stress(1) - p_stress(2))^2 + (p_stress(1) - p_stress(3))^2 + (p_stress(3) - p_stress(2))^2))