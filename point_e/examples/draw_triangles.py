import numpy as np
from wand.image import Image
from wand.color import Color
from wand.drawing import Drawing
from wand.display import display

# define vertices of triangle
p1 = (250, 100)
p2 = (100, 400)
p3 = (400, 400)

# define barycentric colors and vertices
colors = {
    Color('RED'): p1,
    Color('GREEN1'): p2,
    Color('BLUE'): p3
}

# create black image
black = np.zeros([512, 512, 3], dtype=np.uint8)
def draw_triangles(triangles_coords, triangle_colors):
    with Image.from_array(black) as img:
        # with img.clone() as mask:
        with Drawing() as draw:
            for i in range(len(triangles_coords)):
                # draw.polygon(triangles_coords[i], fill=triangle_colors[i])
                points = triangles_coords[i]
                current_colors = triangle_colors[i]
                # sort colors
                sorted_colors = sorted(current_colors, key=lambda x: x[1])
                # take median color
                median_color = sorted_colors[1]
                # average_color = [(current_colors[0][0] + current_colors[1][0] + current_colors[2][0]) / 3,
                #                     (current_colors[0][1] + current_colors[1][1] + current_colors[2][1]) / 3,
                #                     (current_colors[0][2] + current_colors[1][2] + current_colors[2][2]) / 3]
                draw.fill_color = Color(f'rgb({median_color[0]}, {median_color[1]}, {median_color[2]})')
                draw.polygon([(points[0][0], points[0][1]), (points[1][0], points[1][1]), (points[2][0], points[2][1])])
                draw.draw(img)

                # todo lerp colors
                # draw.fill_color = Color('white')
                # draw.polygon(points)
                # draw.draw(mask)
                # colors = {
                #
                #     Color(f'rgb({current_colors[0][0]}, {current_colors[0][1]}, {current_colors[0][2]})'): points[0],
                #     Color(f'rgb({current_colors[1][0]}, {current_colors[1][1]}, {current_colors[1][2]})'): points[1],
                #     Color(f'rgb({current_colors[2][0]}, {current_colors[2][1]}, {current_colors[2][2]})'): points[2]
                # }
                # img.sparse_color('barycentric', colors)
                # img.composite_channel('all_channels', mask, 'atop', 0, 0)
            img.format = 'png'
            img.save(filename='barycentric_image.png')
            # display(img)
        img.format = 'rgb'
        img.alpha_channel = False
        # img.rotate(270)
        return np.asarray(bytearray(img.make_blob()), dtype=np.uint8).reshape(512,512,3)

#v3do your_mesh.fillHoles(size=size_to_fill)

if __name__ == '__main__':
    triangles_coords = [[p1, p2, p3], [(100,0), (0,0), (500,500)]]
    triangle_colors = [[[200,200,200], [200,200,200], [200,200,200]], [[255,255,100], [255,100,100], [255,100,100]]]
    # draw_triangles(triangles_coords, triangle_colors)
    triangles_coords += [[(0,0), (256,256), (0,256)], [(100,0), (0,0), (500,500)]]
    triangle_colors += [[[200,200,200], [200,200,200], [200,200,200]], [[255,255,100], [255,100,100], [255,100,100]]]
    draw_triangles(triangles_coords, triangle_colors)
