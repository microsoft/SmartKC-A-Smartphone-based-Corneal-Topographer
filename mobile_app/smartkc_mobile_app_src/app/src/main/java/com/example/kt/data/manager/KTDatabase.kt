/**
 * Database class: Creates new singleton database object
 */
package com.example.kt.data.manager

import android.content.Context
import androidx.room.AutoMigration
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import com.example.kt.data.local.daos.FileDao
import com.example.kt.data.models.KTFile

@Database(
  entities =
    [
      KTFile::class
    ],
  version = 2,
  autoMigrations = [AutoMigration(from = 1, to = 2)],
  exportSchema = true
)

abstract class KTDatabase : RoomDatabase() {
  abstract fun fileDao(): FileDao


  companion object {
    private var INSTANCE: KTDatabase? = null

    fun getInstance(context: Context): KTDatabase? {
      if (INSTANCE == null) {
        synchronized(KTDatabase::class) {
          INSTANCE = Room.databaseBuilder(context.applicationContext, KTDatabase::class.java, "kt.db").build()
        }
      }
      return INSTANCE
    }
  }
}
