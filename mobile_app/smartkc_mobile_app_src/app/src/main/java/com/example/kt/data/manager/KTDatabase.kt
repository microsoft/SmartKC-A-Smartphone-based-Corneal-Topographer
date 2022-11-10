// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

package com.example.kt.data.manager

import android.content.Context
import androidx.room.AutoMigration
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import com.microsoft.research.karya.data.local.Converters
import com.microsoft.research.karya.data.local.daos.*
import com.microsoft.research.karya.data.local.daosExtra.MicrotaskAssignmentDaoExtra
import com.microsoft.research.karya.data.local.daosExtra.MicrotaskDaoExtra
import com.microsoft.research.karya.data.model.karya.*

@Database(
  entities =
    [
      WorkerRecord::class,
      KaryaFileRecord::class,
      TaskRecord::class,
      MicroTaskRecord::class,
      MicroTaskAssignmentRecord::class,
      PaymentAccountRecord::class,
      LeaderboardRecord::class
    ],
  version = 2,
  autoMigrations = [AutoMigration(from = 1, to = 2)],
  exportSchema = true
)
@TypeConverters(Converters::class)
abstract class KaryaDatabase : RoomDatabase() {
  abstract fun microTaskDao(): MicroTaskDao
  abstract fun taskDao(): TaskDao
  abstract fun workerDao(): WorkerDao
  abstract fun microtaskAssignmentDao(): MicroTaskAssignmentDao

  abstract fun microtaskAssignmentDaoExtra(): MicrotaskAssignmentDaoExtra
  abstract fun microtaskDaoExtra(): MicrotaskDaoExtra
  abstract fun karyaFileDao(): KaryaFileDao
  abstract fun paymentAccountDao(): PaymentAccountDao
  abstract fun leaderboardDao(): LeaderboardDao

  companion object {
    private var INSTANCE: KaryaDatabase? = null

    fun getInstance(context: Context): KaryaDatabase? {
      if (INSTANCE == null) {
        synchronized(KaryaDatabase::class) {
          INSTANCE = Room.databaseBuilder(context.applicationContext, KaryaDatabase::class.java, "karya.db").build()
        }
      }
      return INSTANCE
    }
  }
}
