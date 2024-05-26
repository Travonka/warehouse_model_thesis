import datetime as dt
import itertools
import inspect
import logging
import matplotlib.pyplot as plt
import operator
import os
import pandas as pd
import pickle
import random
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import json
from random import shuffle
from src.shelf import Shelf

class Warehouse:
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.warehouse_map = [[None] * columns for _ in range(rows)]

    def add_shelf(self, row, column, max_volume_liters):
        if row < self.rows and column < self.columns:
            self.warehouse_map[row][column] = Shelf(max_volume_liters)

    def remove_shelf(self, row, column):
        if row < self.rows and column < self.columns:
            self.warehouse_map[row][column] = None

    def visualize_warehouse(self, visualization_type):
        if visualization_type not in ['occupancy', 'idc', 'coi']:
            raise ValueError("Invalid visualization type. Must be one of: 'occupancy', 'idc', 'coi'")

        warehouse_matrix = np.zeros((self.rows, self.columns))
        for i in range(self.rows):
            for j in range(self.columns):
                if self.warehouse_map[i][j]:
                    if visualization_type == 'occupancy':
                        warehouse_matrix[i][j] = 1 - self.warehouse_map[i][j].occupancy_ratio()
                    elif visualization_type == 'idc':
                        if self.warehouse_map[i][j].products:
                            avg_idc = sum([product.idc for product in self.warehouse_map[i][j].products]) / len(self.warehouse_map[i][j].products)
                            warehouse_matrix[i][j] = avg_idc
                        else:
                            warehouse_matrix[i][j] = -1
                    elif visualization_type == 'coi':
                        if self.warehouse_map[i][j].products:
                            avg_coi = sum([product.coi for product in self.warehouse_map[i][j].products]) / len(self.warehouse_map[i][j].products)
                            warehouse_matrix[i][j] = avg_coi
                        else:
                            warehouse_matrix[i][j] = -1
                else:
                    warehouse_matrix[i][j] = np.nan

        plt.figure(figsize=(20,20))
        
        if visualization_type == 'occupancy':
            plt.imshow(warehouse_matrix, cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
            plt.colorbar(label='Occupancy Ratio')
        elif visualization_type in ['idc', 'coi']:
            plt.imshow(warehouse_matrix, cmap='RdYlGn', interpolation='nearest')
            plt.colorbar(label='Average Value')
        plt.title('Warehouse Contents')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.xticks(range(self.columns))
        plt.yticks(range(self.rows))
        plt.grid(True)
        plt.show()
        
    def find_closest_item_to_point(self, itemId, row, column, routing_policy):
        min_distance = float('inf')
        nearest_product_coordinates = None
        
        for i in range(self.rows):
            for j in range(self.columns):
                if self.warehouse_map[i][j] and self.warehouse_map[i][j].products:
                    for product in self.warehouse_map[i][j].products:
                        if product.id == itemId:
                            distance = abs(row - i) + abs(column - j)
                            if distance < min_distance:
                                min_distance = distance
                                nearest_product_coordinates = (i, j)
        
        return nearest_product_coordinates

    
    def visualize_product_location(self, products):
            locations = list()
            for item in products:
                location = self.find_product_location(item.id, item.exemplar_id)
                locations.append(location)

            print(locations)
            warehouse_matrix = np.zeros((self.rows, self.columns))

            for i in range(self.rows):
                for j in range(self.columns):
                    if self.warehouse_map[i][j]:
                        if (i, j) in locations:
                            warehouse_matrix[i][j] = 100
                        else:
                            warehouse_matrix[i][j] = 0
                    else:
                        warehouse_matrix[i][j] = np.nan

            plt.figure(figsize=(20,20))
        
            plt.imshow(warehouse_matrix, cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=100)
            plt.colorbar(label='Occupancy Ratio')
            plt.title('Warehouse Contents')
            plt.xlabel('Columns')
            plt.ylabel('Rows')
            plt.xticks(range(self.columns))
            plt.yticks(range(self.rows))
            plt.grid(True)
            plt.show()
            
    def visualize_specific_locations(self, locations):
            warehouse_matrix = np.zeros((self.rows, self.columns))

            for i in range(self.rows):
                for j in range(self.columns):
                    if self.warehouse_map[i][j]:
                        if (i, j) in locations:
                            warehouse_matrix[i][j] = 100
                        else:
                            warehouse_matrix[i][j] = 0
                    else:
                        warehouse_matrix[i][j] = np.nan

            plt.figure(figsize=(20,20))
            plt.imshow(warehouse_matrix, cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=100)
            plt.colorbar(label='Occupancy Ratio')
            plt.title('Warehouse Contents')
            plt.xlabel('Columns')
            plt.ylabel('Rows')
            plt.xticks(range(self.columns))
            plt.yticks(range(self.rows))
            plt.grid(True)
            plt.show()

    def find_product_location(self, item_id, exemplar_id):
        for i in range(self.rows):
            for j in range(self.columns):
                if self.warehouse_map[i][j] and self.warehouse_map[i][j].products:
                    for product in self.warehouse_map[i][j].products:
                        if product.id == item_id and product.exemplar_id == exemplar_id:
                            return i, j 
        return None 
    
    def visualize_picking_route(self, route, item_locations):
        warehouse_matrix = np.zeros((self.rows, self.columns))
        plt.figure(figsize=(20,20))
        for i in range(len(route) - 1):
            current_location = route[i]
            next_location = route[i + 1]
            x1, y1 = current_location
            x2, y2 = next_location
            plt.plot([y1, y2], [x1, x2], color='blue', linewidth=2)

        for i in range(self.rows):
            for j in range(self.columns):
                if self.warehouse_map[i][j]:                    
                    warehouse_matrix[i][j] = 0
                    if (i, j) in item_locations:
                        warehouse_matrix[i][j] = 100
                else:
                    warehouse_matrix[i][j] = np.nan
                    
        plt.imshow(warehouse_matrix, cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=100)
        plt.colorbar(label='Occupancy Ratio')
        plt.title('Picking Route')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.xticks(range(self.columns))
        plt.yticks(range(self.rows))
        plt.grid(True)
        plt.show()

    def L1_distance(self, point1, point2): ## L1 dist
        x1, y1 = point1
        x2, y2 = point2
        return abs(x2 - x1) + abs(y2 - y1)
    
    def perform_order_pick(self, routing_policy, locations_with_items, visualize):
        counter = 0
        route = []  
        locations = [o[0] for o in locations_with_items]
        for product_with_locations in locations_with_items:
            product = product_with_locations[1]
            location = product_with_locations[0]
            self.warehouse_map[location[0]][location[1]].remove_product(product)

        if(routing_policy == 's_shape'):
            locations = sorted(locations, key = lambda x: x[1])
            route.append((0,0))
            visited_columns = set()
            on_other_side = False
            for location in locations:
                target_column = location[1]
                if target_column in visited_columns:
                    continue
                visited_columns.add(target_column)
                if(self.warehouse_map[2][target_column+1] is not None):
                    visited_columns.add(target_column - 2)
                    target_column = target_column - 1
                else:
                    target_column = target_column + 1
                    visited_columns.add(target_column + 2)

                if(on_other_side):
                    route.append((18, target_column))
                    route.append((0, target_column))
                    on_other_side = False
                else:
                    route.append((0, target_column))
                    route.append((18, target_column))
                    on_other_side = True
            
            if(on_other_side):
                col = locations[len(locations)-1][1]
                if(self.warehouse_map[2][col+1] is not None):
                    col = col - 1
                else:
                    col = col + 1
                route.append((18, col))
                route.append((18, 0))
                
            route.append((0, 0))
            if(visualize == True):
                print(route)  
                self.visualize_picking_route(route, locations)
            for i in range(len(route) - 1):
                counter += self.L1_distance(route[i], route[i+1]) 
            return counter

        if(routing_policy == 'return'): 
            locations = sorted(locations, key = lambda x: x[1])
            route.append((0,0))
            visited_columns = set()
            for location in locations:
                target_column = location[1]
                if target_column in visited_columns:
                    continue
                visited_columns.add(target_column)
                max_x = float('-inf') 
                if(self.warehouse_map[2][target_column+1] is not None):
                    visited_columns.add(target_column - 2)
                    target_column = target_column - 1
                    for x, y in locations:
                        if y == target_column + 1 and x > max_x:
                            max_x = x
                else:
                    target_column = target_column + 1
                    visited_columns.add(target_column + 2)
                    for x, y in locations:
                        if y == target_column - 1 and x > max_x:
                            max_x = x

                for x, y in locations:
                    if y == target_column and x > max_x:
                        max_x = x

                route.append((0, target_column))
                route.append((max_x, target_column))
                route.append((0, target_column))
            
            col = locations[len(locations)-1][1]
            if(self.warehouse_map[2][col+1] is not None):
                col = col - 1
            else:
                col = col + 1
            route.append((0, col))
            route.append((0,0))
            if(visualize == True):
                print(route)  
                self.visualize_picking_route(route, locations)
            for i in range(len(route) - 1):
                counter += self.L1_distance(route[i], route[i+1]) 

        return counter

    def place_product(self, product, routing_policy):
        if(routing_policy == 's_shape'):
            for col in range(1,41):
                for row in range(2, 18):
                    if(col % 3 != 0):
                        if self.warehouse_map[row][col] is not None and self.warehouse_map[row][col].check_fit(product):
                            shelf = self.warehouse_map[row][col]
                            shelf.add_product(product)
                            return
        elif(routing_policy == 'return'):
            for row in range(2,18):
                for col in range(1, 41):
                    if(col % 3 != 0):
                        if self.warehouse_map[row][col] is not None and self.warehouse_map[row][col].check_fit(product):
                            shelf = self.warehouse_map[row][col]
                            shelf.add_product(product)
                            return
        elif(routing_policy == 'random'):
            while(True):
                row = random.randint(2, 18)
                col = random.randint(1, 41)
                if(col % 3 != 0):
                    if self.warehouse_map[row][col] is not None and self.warehouse_map[row][col].check_fit(product):
                        shelf = self.warehouse_map[row][col]
                        shelf.add_product(product)
                        return           